import argparse
import csv
import json
from pathlib import Path
from uuid import uuid4

from confluent_kafka import KafkaError, Message

from fraud_detection.messaging import create_producer


def publish(
    path: Path,
    limit: int,
) -> None:
    producer = create_producer()
    errors = []

    def on_delivery(
        error: KafkaError | None,
        message: Message,
    ) -> None:
        if error is not None:
            errors.append(str(error))

    count = 0
    with path.open(newline='', encoding='utf-8-sig') as stream:
        for row in csv.DictReader(stream):
            if count >= limit:
                break
            transaction = dict(row)
            for field in (
                'lat',
                'lon',
                'merchant_lat',
                'merchant_lon',
                'amount',
                'population_city',
            ):
                transaction[field] = float(row[field])
            transaction['cat_id'] = row['cat_id'] or None
            transaction_id = str(uuid4())
            payload = json.dumps(
                {'transaction_id': transaction_id, 'data': transaction}, allow_nan=False
            ).encode()
            while True:
                try:
                    producer.produce(
                        'transactions',
                        key=transaction_id,
                        value=payload,
                        on_delivery=on_delivery,
                    )
                    break
                except BufferError:
                    producer.poll(0.1)
            producer.poll(0)
            count += 1
            if errors:
                raise RuntimeError(errors[0])
    remaining = producer.flush(15)
    if errors or remaining:
        raise RuntimeError(errors[0] if errors else 'Kafka delivery timed out')
    print(f'Sent {count} transactions')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=100)
    args = parser.parse_args()
    if args.limit < 1:
        parser.error('--limit must be positive')
    publish(args.csv, args.limit)


if __name__ == '__main__':
    main()
