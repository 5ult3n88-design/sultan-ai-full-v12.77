"""
Sultan AI Model Monitor
=======================
Track model performance and trigger retraining
"""

import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pickle

MONITOR_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'monitor')
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')


class ModelMonitor:
    """Monitor model performance over time"""

    def __init__(self):
        os.makedirs(MONITOR_DIR, exist_ok=True)
        self.metrics_file = os.path.join(MONITOR_DIR, 'model_metrics.json')
        self.alerts_file = os.path.join(MONITOR_DIR, 'alerts.json')
        self.metrics = self._load_metrics()
        self.alerts = self._load_alerts()

    def _load_metrics(self) -> Dict:
        if os.path.exists(self.metrics_file):
            try:
                with open(self.metrics_file, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}

    def _load_alerts(self) -> List:
        if os.path.exists(self.alerts_file):
            try:
                with open(self.alerts_file, 'r') as f:
                    return json.load(f)
            except:
                return []
        return []

    def _save_metrics(self):
        with open(self.metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2, default=str)

    def _save_alerts(self):
        with open(self.alerts_file, 'w') as f:
            json.dump(self.alerts, f, indent=2, default=str)

    def log_prediction(self, symbol: str, prediction: Dict, actual_outcome: Optional[float] = None):
        """Log a prediction for monitoring"""
        if symbol not in self.metrics:
            self.metrics[symbol] = {
                'predictions': [],
                'accuracy_history': [],
                'last_retrain': None
            }

        entry = {
            'timestamp': datetime.now().isoformat(),
            'direction': prediction.get('direction'),
            'confidence': prediction.get('confidence'),
            'method': prediction.get('method'),
            'actual': actual_outcome
        }

        self.metrics[symbol]['predictions'].append(entry)

        # Keep only last 1000 predictions
        if len(self.metrics[symbol]['predictions']) > 1000:
            self.metrics[symbol]['predictions'] = self.metrics[symbol]['predictions'][-1000:]

        self._save_metrics()

    def update_outcome(self, symbol: str, timestamp: str, actual_direction: str, was_correct: bool):
        """Update prediction with actual outcome"""
        if symbol not in self.metrics:
            return

        for pred in self.metrics[symbol]['predictions']:
            if pred['timestamp'] == timestamp:
                pred['actual'] = actual_direction
                pred['correct'] = was_correct
                break

        self._save_metrics()
        self._check_performance(symbol)

    def _check_performance(self, symbol: str):
        """Check if model performance is degrading"""
        if symbol not in self.metrics:
            return

        recent = [p for p in self.metrics[symbol]['predictions']
                  if p.get('correct') is not None][-50:]

        if len(recent) < 20:
            return

        accuracy = sum(1 for p in recent if p['correct']) / len(recent)

        # Log accuracy history
        self.metrics[symbol]['accuracy_history'].append({
            'timestamp': datetime.now().isoformat(),
            'accuracy': accuracy,
            'sample_size': len(recent)
        })

        # Keep only last 100 accuracy records
        if len(self.metrics[symbol]['accuracy_history']) > 100:
            self.metrics[symbol]['accuracy_history'] = \
                self.metrics[symbol]['accuracy_history'][-100:]

        # Alert if accuracy drops below 45%
        if accuracy < 0.45:
            self._add_alert(symbol, 'low_accuracy',
                          f"Model accuracy dropped to {accuracy:.1%}")

        self._save_metrics()

    def _add_alert(self, symbol: str, alert_type: str, message: str):
        """Add an alert"""
        alert = {
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'type': alert_type,
            'message': message,
            'resolved': False
        }
        self.alerts.append(alert)
        self._save_alerts()

    def get_model_health(self, symbol: str) -> Dict:
        """Get model health status"""
        if symbol not in self.metrics:
            return {'status': 'unknown', 'accuracy': None}

        history = self.metrics[symbol].get('accuracy_history', [])
        if not history:
            return {'status': 'no_data', 'accuracy': None}

        latest = history[-1]['accuracy']
        avg_recent = sum(h['accuracy'] for h in history[-10:]) / min(len(history), 10)

        if latest >= 0.55:
            status = 'healthy'
        elif latest >= 0.48:
            status = 'warning'
        else:
            status = 'critical'

        return {
            'status': status,
            'accuracy': latest,
            'avg_accuracy': avg_recent,
            'trend': 'improving' if len(history) > 1 and latest > history[-2]['accuracy'] else 'declining',
            'last_updated': history[-1]['timestamp']
        }

    def needs_retrain(self, symbol: str) -> bool:
        """Check if model needs retraining"""
        if symbol not in self.metrics:
            return True

        last_retrain = self.metrics[symbol].get('last_retrain')
        if not last_retrain:
            return True

        # Retrain if older than 7 days
        try:
            last_dt = datetime.fromisoformat(last_retrain)
            if datetime.now() - last_dt > timedelta(days=7):
                return True
        except:
            return True

        # Retrain if accuracy is below 45%
        health = self.get_model_health(symbol)
        if health.get('accuracy') and health['accuracy'] < 0.45:
            return True

        return False

    def mark_retrained(self, symbol: str):
        """Mark model as retrained"""
        if symbol not in self.metrics:
            self.metrics[symbol] = {'predictions': [], 'accuracy_history': []}

        self.metrics[symbol]['last_retrain'] = datetime.now().isoformat()
        self._save_metrics()

    def get_active_alerts(self) -> List[Dict]:
        """Get unresolved alerts"""
        return [a for a in self.alerts if not a['resolved']]

    def get_dashboard_summary(self) -> Dict:
        """Get summary for dashboard display"""
        summary = {
            'models': {},
            'total_predictions': 0,
            'avg_accuracy': 0,
            'alerts': len(self.get_active_alerts())
        }

        accuracies = []
        for symbol, data in self.metrics.items():
            health = self.get_model_health(symbol)
            summary['models'][symbol] = health
            summary['total_predictions'] += len(data.get('predictions', []))
            if health.get('accuracy'):
                accuracies.append(health['accuracy'])

        if accuracies:
            summary['avg_accuracy'] = sum(accuracies) / len(accuracies)

        return summary


# Global monitor instance
_monitor = None


def get_monitor() -> ModelMonitor:
    """Get singleton monitor instance"""
    global _monitor
    if _monitor is None:
        _monitor = ModelMonitor()
    return _monitor
