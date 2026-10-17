import json
import logging
import os
from uuid import UUID

from confluent_kafka import KafkaException, Message

from fraud_detection.messaging import create_consumer, create_producer, publish_many
from fraud_detection.scorer import Scorer


def score_messages(
    scorer: Scorer,
    messages: list[Message],
) -> list[dict]:
    ids = []
    transactions = []
    for message in messages:
        if message.error():
            raise KafkaException(message.error())
        try:
            payload = json.loads(message.value())
            if not isinstance(payload['transaction_id'], str):
                raise ValueError('Transaction id must be a string')
            transaction_id = str(UUID(payload['transaction_id']))
            data = payload['data']
            if not isinstance(data, dict):
                raise ValueError('Transaction data must be an object')
        except (ValueError, TypeError, KeyError) as error:
            logging.warning('Invalid transaction: %s', error)
            continue
        ids.append(transaction_id)
        transactions.append(data)
    if not transactions:
        return []
    try:
        predictions = scorer.score_many(transactions)
    except (ValueError, TypeError, KeyError):
        valid_ids = []
        predictions = []
        for transaction_id, data in zip(ids, transactions, strict=True):
            try:
                prediction = scorer.score(data)
            except (ValueError, TypeError, KeyError) as error:
                logging.warning('Invalid transaction: %s', error)
                continue
            valid_ids.append(transaction_id)
            predictions.append(prediction)
        ids = valid_ids
    return [
        {
            'transaction_id': transaction_id,
            'score': score,
            'fraud_flag': flag,
            'model_version': version,
        }
        for transaction_id, (score, flag, version) in zip(ids, predictions, strict=True)
    ]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
    scorer = Scorer(threshold=float(os.getenv('FRAUD_THRESHOLD', '0.98')))
    consumer = create_consumer('fraud-scorer', 'transactions')
    producer = create_producer()
    logging.info('Scoring service ready')
    try:
        while True:
            messages = consumer.consume(100, timeout=0.2)
            if not messages:
                continue
            results = score_messages(scorer, messages)
            publish_many(producer, 'scores', results)
            consumer.commit(asynchronous=False)
            logging.info('Scored %s transactions', len(results))
    finally:
        consumer.close()


if __name__ == '__main__':
    main()
