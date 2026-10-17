import json
import os
from typing import Any

from confluent_kafka import Consumer, KafkaError, Message, Producer


class DeliveryError(RuntimeError):
    def __init__(
        self,
        message: str,
        delivered_ids: list[str],
    ) -> None:
        super().__init__(message)
        self.delivered_ids = delivered_ids


def create_consumer(
    group: str,
    topic: str,
) -> Consumer:
    consumer = Consumer(
        {
            'bootstrap.servers': os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'kafka:9092'),
            'group.id': group,
            'auto.offset.reset': 'earliest',
            'enable.auto.commit': False,
        }
    )
    consumer.subscribe([topic])
    return consumer


def create_producer() -> Producer:
    return Producer(
        {
            'bootstrap.servers': os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'kafka:9092'),
            'enable.idempotence': True,
            'delivery.timeout.ms': 10000,
        }
    )


def publish(
    producer: Producer,
    topic: str,
    message: dict[str, Any],
) -> None:
    publish_many(producer, topic, [message])


def publish_many(
    producer: Producer,
    topic: str,
    messages: list[dict[str, Any]],
) -> None:
    errors: list[KafkaError] = []
    delivered_ids: list[str] = []

    def record_delivery(
        error: KafkaError | None,
        message: Message,
    ) -> None:
        if error:
            errors.append(error)
        else:
            delivered_ids.append(message.key().decode())

    try:
        for message in messages:
            producer.produce(
                topic,
                key=str(message['transaction_id']),
                value=json.dumps(message, allow_nan=False).encode(),
                on_delivery=record_delivery,
            )
    except Exception as error:
        producer.flush(15)
        raise DeliveryError(str(error), delivered_ids) from error
    if producer.flush(15):
        raise DeliveryError('Kafka delivery timed out', delivered_ids)
    if errors:
        raise DeliveryError(str(errors[0]), delivered_ids)
