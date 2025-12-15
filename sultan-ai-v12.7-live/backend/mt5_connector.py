"""
MetaTrader 5 Connector for Trading Robot
Connects the autonomous trading robot to MT5 for live/demo trading
"""

import MetaTrader5 as mt5
from datetime import datetime
import pandas as pd
import time

class MT5Connector:
    """Connect trading robot to MetaTrader 5"""

    def __init__(self, account=None, password=None, server=None, path=None):
        """
        Initialize MT5 connection

        Args:
            account: MT5 account number (optional if already logged in)
            password: Account password
            server: Broker server name
            path: Path to MT5 terminal (optional, auto-detected on Windows)
        """
        self.account = account
        self.password = password
        self.server = server
        self.path = path
        self.connected = False
        self.positions = {}

    def connect(self):
        """Connect to MT5 terminal"""
        try:
            # Initialize MT5
            if self.path:
                if not mt5.initialize(path=self.path):
                    print(f"MT5 initialize() failed, error code: {mt5.last_error()}")
                    return False
            else:
                if not mt5.initialize():
                    print(f"MT5 initialize() failed, error code: {mt5.last_error()}")
                    return False

            # Login if credentials provided
            if self.account and self.password and self.server:
                authorized = mt5.login(
                    login=self.account,
                    password=self.password,
                    server=self.server
                )

                if not authorized:
                    print(f"Login failed, error code: {mt5.last_error()}")
                    mt5.shutdown()
                    return False

            # Get account info
            account_info = mt5.account_info()
            if account_info is None:
                print("Failed to get account info")
                mt5.shutdown()
                return False

            self.connected = True

            print(f"✅ Connected to MT5")
            print(f"   Account: {account_info.login}")
            print(f"   Server: {account_info.server}")
            print(f"   Balance: ${account_info.balance:.2f}")
            print(f"   Equity: ${account_info.equity:.2f}")
            print(f"   Leverage: 1:{account_info.leverage}")

            return True

        except Exception as e:
            print(f"❌ Connection error: {e}")
            return False

    def disconnect(self):
        """Disconnect from MT5"""
        if self.connected:
            mt5.shutdown()
            self.connected = False
            print("Disconnected from MT5")

    def get_account_info(self):
        """Get account information"""
        if not self.connected:
            return None

        account_info = mt5.account_info()
        if account_info is None:
            return None

        return {
            'login': account_info.login,
            'server': account_info.server,
            'balance': account_info.balance,
            'equity': account_info.equity,
            'margin': account_info.margin,
            'margin_free': account_info.margin_free,
            'profit': account_info.profit,
            'leverage': account_info.leverage,
            'currency': account_info.currency
        }

    def get_symbol_info(self, symbol):
        """Get symbol information and normalize symbol name"""
        if not self.connected:
            return None

        # Try to find symbol (MT5 symbols might have suffixes like .raw, .ecn, etc.)
        symbol_info = mt5.symbol_info(symbol)

        if symbol_info is None:
            # Try common variations
            variations = [
                symbol,
                symbol.replace('=X', ''),  # EURUSD=X -> EURUSD
                symbol.replace('USD', ''),  # For forex pairs
                symbol + '.raw',
                symbol + '.ecn',
                symbol + 'm',
            ]

            for var in variations:
                symbol_info = mt5.symbol_info(var)
                if symbol_info is not None:
                    print(f"Found symbol as: {var}")
                    symbol = var
                    break

        if symbol_info is None:
            print(f"Symbol {symbol} not found")
            return None

        # Enable symbol in Market Watch if needed
        if not symbol_info.visible:
            if not mt5.symbol_select(symbol, True):
                print(f"Failed to select symbol {symbol}")
                return None

        return {
            'name': symbol_info.name,
            'bid': symbol_info.bid,
            'ask': symbol_info.ask,
            'spread': symbol_info.spread,
            'digits': symbol_info.digits,
            'point': symbol_info.point,
            'trade_contract_size': symbol_info.trade_contract_size,
            'volume_min': symbol_info.volume_min,
            'volume_max': symbol_info.volume_max,
            'volume_step': symbol_info.volume_step,
        }

    def calculate_lot_size(self, symbol, risk_amount, stop_loss_pips):
        """Calculate lot size based on risk amount and stop loss"""
        symbol_info = self.get_symbol_info(symbol)
        if symbol_info is None:
            return None

        point = symbol_info['point']
        contract_size = symbol_info['trade_contract_size']

        # Calculate lot size
        # Risk = Lot Size * Contract Size * Stop Loss in Points * Point Value
        pip_value = contract_size * point
        stop_loss_points = stop_loss_pips / point if point > 0 else 1

        lot_size = risk_amount / (stop_loss_points * pip_value) if pip_value > 0 else 0.01

        # Round to volume step
        volume_step = symbol_info['volume_step']
        lot_size = round(lot_size / volume_step) * volume_step

        # Clamp to min/max
        lot_size = max(symbol_info['volume_min'], min(lot_size, symbol_info['volume_max']))

        return lot_size

    def open_position(self, symbol, action, volume=None, risk_amount=100,
                     stop_loss=None, take_profit=None, comment="Robot"):
        """
        Open a trading position

        Args:
            symbol: Trading symbol
            action: 'BUY' or 'SELL'
            volume: Lot size (if None, calculated from risk_amount)
            risk_amount: Risk amount in account currency
            stop_loss: Stop loss price
            take_profit: Take profit price
            comment: Order comment
        """
        if not self.connected:
            return False, "Not connected to MT5"

        # Get symbol info
        symbol_info = self.get_symbol_info(symbol)
        if symbol_info is None:
            return False, f"Symbol {symbol} not available"

        symbol = symbol_info['name']  # Use the actual MT5 symbol name

        # Calculate lot size if not provided
        if volume is None:
            if stop_loss:
                current_price = symbol_info['bid'] if action == 'SELL' else symbol_info['ask']
                stop_loss_pips = abs(current_price - stop_loss) / symbol_info['point']
                volume = self.calculate_lot_size(symbol, risk_amount, stop_loss_pips)
            else:
                volume = symbol_info['volume_min']

        # Prepare order request
        order_type = mt5.ORDER_TYPE_BUY if action == 'BUY' else mt5.ORDER_TYPE_SELL
        price = symbol_info['ask'] if action == 'BUY' else symbol_info['bid']

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": order_type,
            "price": price,
            "deviation": 20,
            "magic": 234000,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        # Add stop loss if provided
        if stop_loss:
            request["sl"] = stop_loss

        # Add take profit if provided
        if take_profit:
            request["tp"] = take_profit

        # Send order
        result = mt5.order_send(request)

        if result is None:
            return False, f"Order send failed: {mt5.last_error()}"

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            return False, f"Order failed: {result.comment} (code: {result.retcode})"

        # Store position info
        self.positions[result.order] = {
            'ticket': result.order,
            'symbol': symbol,
            'action': action,
            'volume': volume,
            'open_price': result.price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'open_time': datetime.now(),
            'comment': comment
        }

        return True, f"Position opened: {action} {volume} lots of {symbol} at {result.price}"

    def close_position(self, ticket=None, symbol=None):
        """Close a position by ticket or symbol"""
        if not self.connected:
            return False, "Not connected to MT5"

        # Get position
        if ticket:
            position = mt5.positions_get(ticket=ticket)
        elif symbol:
            position = mt5.positions_get(symbol=symbol)
        else:
            return False, "Must provide ticket or symbol"

        if position is None or len(position) == 0:
            return False, "Position not found"

        position = position[0]

        # Prepare close request
        order_type = mt5.ORDER_TYPE_SELL if position.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
        price = mt5.symbol_info_tick(position.symbol).bid if position.type == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(position.symbol).ask

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": position.symbol,
            "volume": position.volume,
            "type": order_type,
            "position": position.ticket,
            "price": price,
            "deviation": 20,
            "magic": 234000,
            "comment": "Close by Robot",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        # Send close order
        result = mt5.order_send(request)

        if result is None:
            return False, f"Close order failed: {mt5.last_error()}"

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            return False, f"Close failed: {result.comment} (code: {result.retcode})"

        # Remove from tracking
        if position.ticket in self.positions:
            del self.positions[position.ticket]

        profit = position.profit
        return True, f"Position closed with profit: ${profit:.2f}"

    def get_open_positions(self):
        """Get all open positions"""
        if not self.connected:
            return []

        positions = mt5.positions_get()
        if positions is None:
            return []

        result = []
        for pos in positions:
            result.append({
                'ticket': pos.ticket,
                'symbol': pos.symbol,
                'type': 'BUY' if pos.type == mt5.ORDER_TYPE_BUY else 'SELL',
                'volume': pos.volume,
                'open_price': pos.price_open,
                'current_price': pos.price_current,
                'stop_loss': pos.sl,
                'take_profit': pos.tp,
                'profit': pos.profit,
                'swap': pos.swap,
                'comment': pos.comment,
                'open_time': datetime.fromtimestamp(pos.time)
            })

        return result

    def get_historical_data(self, symbol, timeframe='1H', bars=1000):
        """Get historical price data"""
        if not self.connected:
            return None

        # Map timeframe string to MT5 constant
        timeframe_map = {
            '1M': mt5.TIMEFRAME_M1,
            '5M': mt5.TIMEFRAME_M5,
            '15M': mt5.TIMEFRAME_M15,
            '30M': mt5.TIMEFRAME_M30,
            '1H': mt5.TIMEFRAME_H1,
            '4H': mt5.TIMEFRAME_H4,
            '1D': mt5.TIMEFRAME_D1,
        }

        mt5_timeframe = timeframe_map.get(timeframe, mt5.TIMEFRAME_H1)

        # Get symbol info
        symbol_info = self.get_symbol_info(symbol)
        if symbol_info is None:
            return None

        symbol = symbol_info['name']

        # Get rates
        rates = mt5.copy_rates_from_pos(symbol, mt5_timeframe, 0, bars)

        if rates is None:
            print(f"Failed to get rates: {mt5.last_error()}")
            return None

        # Convert to DataFrame
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)

        # Rename columns to match our format
        df.rename(columns={
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'tick_volume': 'Volume'
        }, inplace=True)

        return df[['Open', 'High', 'Low', 'Close', 'Volume']]

    def modify_position(self, ticket, stop_loss=None, take_profit=None):
        """Modify stop loss or take profit of existing position"""
        if not self.connected:
            return False, "Not connected to MT5"

        position = mt5.positions_get(ticket=ticket)
        if position is None or len(position) == 0:
            return False, "Position not found"

        position = position[0]

        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "symbol": position.symbol,
            "position": ticket,
        }

        if stop_loss is not None:
            request["sl"] = stop_loss
        else:
            request["sl"] = position.sl

        if take_profit is not None:
            request["tp"] = take_profit
        else:
            request["tp"] = position.tp

        result = mt5.order_send(request)

        if result is None:
            return False, f"Modify failed: {mt5.last_error()}"

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            return False, f"Modify failed: {result.comment}"

        return True, "Position modified successfully"


def test_connection():
    """Test MT5 connection"""
    print("Testing MT5 Connection...")
    print("-" * 60)

    # Create connector (will use already logged in MT5 terminal)
    mt5_connector = MT5Connector()

    # Connect
    if not mt5_connector.connect():
        print("❌ Failed to connect to MT5")
        print("\nMake sure:")
        print("1. MT5 terminal is installed")
        print("2. MT5 terminal is running")
        print("3. You're logged into a demo/live account")
        return

    # Get account info
    account = mt5_connector.get_account_info()
    if account:
        print("\n✅ Account Information:")
        print(f"   Balance: ${account['balance']:.2f}")
        print(f"   Equity: ${account['equity']:.2f}")
        print(f"   Profit: ${account['profit']:.2f}")
        print(f"   Margin Free: ${account['margin_free']:.2f}")

    # Get open positions
    positions = mt5_connector.get_open_positions()
    print(f"\n📊 Open Positions: {len(positions)}")
    for pos in positions:
        print(f"   {pos['symbol']}: {pos['type']} {pos['volume']} lots, "
              f"Profit: ${pos['profit']:.2f}")

    # Test symbol
    print("\n🔍 Testing Symbol: EURUSD")
    symbol_info = mt5_connector.get_symbol_info('EURUSD')
    if symbol_info:
        print(f"   Bid: {symbol_info['bid']}")
        print(f"   Ask: {symbol_info['ask']}")
        print(f"   Spread: {symbol_info['spread']} points")

    # Disconnect
    mt5_connector.disconnect()
    print("\n✅ Test complete!")


if __name__ == "__main__":
    test_connection()
