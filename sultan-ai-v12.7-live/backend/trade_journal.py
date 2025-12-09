"""
Sultan AI Trade Journal
=======================
Track predictions vs outcomes for model improvement
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional
import pandas as pd

JOURNAL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'journal')


class TradeJournal:
    """Track and analyze trading predictions"""

    def __init__(self, symbol: str):
        self.symbol = symbol
        self.entries: List[Dict] = []
        self.filepath = os.path.join(JOURNAL_DIR, f'{symbol}_journal.json')
        os.makedirs(JOURNAL_DIR, exist_ok=True)
        self._load()

    def _load(self):
        """Load existing journal"""
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, 'r') as f:
                    self.entries = json.load(f)
            except:
                self.entries = []

    def _save(self):
        """Save journal to file"""
        with open(self.filepath, 'w') as f:
            json.dump(self.entries, f, indent=2, default=str)

    def log_prediction(self, prediction: Dict, price_at_prediction: float) -> str:
        """Log a new prediction"""
        entry_id = f"{self.symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        entry = {
            'id': entry_id,
            'symbol': self.symbol,
            'timestamp': datetime.now().isoformat(),
            'price_at_prediction': price_at_prediction,
            'predicted_direction': prediction.get('direction', 'NEUTRAL'),
            'predicted_change': prediction.get('predicted_change', 0),
            'confidence': prediction.get('confidence', 0.5),
            'method': prediction.get('method', 'unknown'),
            'actual_direction': None,
            'actual_change': None,
            'was_correct': None,
            'outcome_logged': False
        }

        self.entries.append(entry)
        self._save()
        return entry_id

    def log_outcome(self, entry_id: str, actual_price: float) -> Optional[Dict]:
        """Log the actual outcome for a prediction"""
        for entry in self.entries:
            if entry['id'] == entry_id and not entry['outcome_logged']:
                price_at_pred = entry['price_at_prediction']
                actual_change = (actual_price - price_at_pred) / price_at_pred

                if actual_change > 0.001:
                    actual_direction = 'UP'
                elif actual_change < -0.001:
                    actual_direction = 'DOWN'
                else:
                    actual_direction = 'NEUTRAL'

                was_correct = (entry['predicted_direction'] == actual_direction) or \
                              (entry['predicted_direction'] == 'NEUTRAL' and abs(actual_change) < 0.005)

                entry['actual_price'] = actual_price
                entry['actual_direction'] = actual_direction
                entry['actual_change'] = actual_change
                entry['was_correct'] = was_correct
                entry['outcome_logged'] = True
                entry['outcome_timestamp'] = datetime.now().isoformat()

                self._save()
                return entry

        return None

    def get_accuracy_stats(self, days: int = 30) -> Dict:
        """Get accuracy statistics for recent predictions"""
        cutoff = datetime.now().timestamp() - (days * 24 * 3600)
        recent = [e for e in self.entries if e['outcome_logged'] and
                  datetime.fromisoformat(e['timestamp']).timestamp() > cutoff]

        if not recent:
            return {'total': 0, 'accuracy': 0, 'by_confidence': {}}

        total = len(recent)
        correct = sum(1 for e in recent if e['was_correct'])

        # Accuracy by confidence level
        by_conf = {}
        for conf_level in ['high', 'medium', 'low']:
            if conf_level == 'high':
                subset = [e for e in recent if e['confidence'] > 0.7]
            elif conf_level == 'medium':
                subset = [e for e in recent if 0.5 <= e['confidence'] <= 0.7]
            else:
                subset = [e for e in recent if e['confidence'] < 0.5]

            if subset:
                by_conf[conf_level] = {
                    'count': len(subset),
                    'correct': sum(1 for e in subset if e['was_correct']),
                    'accuracy': sum(1 for e in subset if e['was_correct']) / len(subset) * 100
                }

        return {
            'total': total,
            'correct': correct,
            'accuracy': correct / total * 100 if total > 0 else 0,
            'by_confidence': by_conf,
            'avg_confidence': sum(e['confidence'] for e in recent) / total if total > 0 else 0
        }

    def get_recent_entries(self, limit: int = 20) -> List[Dict]:
        """Get recent journal entries"""
        return sorted(self.entries, key=lambda x: x['timestamp'], reverse=True)[:limit]


def get_journal(symbol: str) -> TradeJournal:
    """Get or create journal for symbol"""
    return TradeJournal(symbol)


def get_all_journals_stats() -> Dict:
    """Get stats across all symbols"""
    if not os.path.exists(JOURNAL_DIR):
        return {}

    stats = {}
    for f in os.listdir(JOURNAL_DIR):
        if f.endswith('_journal.json'):
            symbol = f.replace('_journal.json', '')
            journal = TradeJournal(symbol)
            stats[symbol] = journal.get_accuracy_stats()

    return stats
