from pathlib import Path
from time import perf_counter
from uuid import uuid4

import numpy as np
import pandas as pd
import streamlit as st

from fraud_detection.database import recent_results, stored_ids
from fraud_detection.messaging import DeliveryError, create_producer, publish_many
from fraud_detection.preprocessing import INPUT_COLUMNS, load_state, preprocess


def send_transactions(
    frame: pd.DataFrame,
) -> None:
    started = perf_counter()
    rows = frame.astype(object).where(pd.notna(frame), None)
    try:
        directory = Path('/runtime/models')
        version = (directory / 'latest.txt').read_text().strip()
        preprocess(rows, load_state(directory / version / 'preprocessing.json'))
    except (ValueError, TypeError, KeyError) as error:
        st.error(f'Некорректные транзакции: {error}')
        return
    ids = [str(uuid4()) for _ in range(len(rows))]
    batch = {
        'pending': ids,
        'total': len(ids),
        'stored': 0,
        'started': started,
        'elapsed': None,
        'error': None,
    }
    st.session_state['batch'] = batch
    confirmed = []
    progress = st.progress(0, text='Отправка в Kafka')
    try:
        producer = create_producer()
        for start in range(0, len(rows), 1000):
            messages = [
                {'transaction_id': transaction_id, 'data': row}
                for transaction_id, row in zip(
                    ids[start : start + 1000],
                    rows.iloc[start : start + 1000].to_dict('records'),
                    strict=True,
                )
            ]
            publish_many(producer, 'transactions', messages)
            confirmed.extend(message['transaction_id'] for message in messages)
            sent = min(start + 1000, len(rows))
            progress.progress(sent / len(rows), text=f'Отправлено: {sent}/{len(rows)}')
    except DeliveryError as error:
        confirmed.extend(error.delivered_ids)
        batch['pending'] = confirmed
        batch['error'] = f'Отправка остановлена: {error}'
    except Exception as error:
        batch['pending'] = confirmed
        batch['error'] = f'Отправка остановлена: {error}'
    st.rerun()


def show_scoring_progress() -> None:
    batch = st.session_state.get('batch')
    if not batch:
        return
    if batch['pending']:
        try:
            stored = stored_ids(batch['pending'])
        except Exception as error:
            st.error(f'Не удалось получить прогресс скоринга: {error}')
            return
        batch['stored'] += len(stored)
        batch['pending'] = [key for key in batch['pending'] if key not in stored]
        if not batch['pending']:
            batch['elapsed'] = perf_counter() - batch['started']
            st.rerun()
    st.progress(
        batch['stored'] / batch['total'],
        text=f'Обработано и сохранено: {batch["stored"]}/{batch["total"]}',
    )
    if batch['error']:
        st.error(batch['error'])
    elif batch['elapsed'] is not None:
        st.success(f'Скоринг завершён за {batch["elapsed"]:.1f} с')


def main() -> None:
    st.set_page_config(page_title='Скоринг транзакций', layout='wide')
    st.title('Скоринг транзакций')
    upload_tab, results_tab = st.tabs(['Отправить транзакции', 'Результаты'])
    with upload_tab:
        file = st.file_uploader('CSV с транзакциями в формате test.csv', type=['csv'])
        if file is not None:
            try:
                frame = pd.read_csv(file)
                missing = set(INPUT_COLUMNS) - set(frame.columns)
                if missing:
                    raise ValueError(
                        f'Отсутствуют столбцы: {", ".join(sorted(missing))}'
                    )
                if frame.empty:
                    raise ValueError('Файл не содержит транзакций')
            except (ValueError, pd.errors.ParserError) as error:
                st.error(str(error))
            else:
                st.write(f'Транзакций в файле: {len(frame)}')
                count = st.number_input(
                    'Сколько транзакций отправить',
                    min_value=1,
                    max_value=len(frame),
                    value=min(100, len(frame)),
                )
                batch = st.session_state.get('batch', {})
                active = bool(batch.get('pending'))
                if st.button('Отправить', disabled=active):
                    send_transactions(frame.head(count))
        batch = st.session_state.get('batch', {})
        active = bool(batch.get('pending'))
        st.fragment(run_every=1 if active else None)(show_scoring_progress)()
    with results_tab:
        if st.button('Посмотреть результаты'):
            try:
                fraud, scores = recent_results()
            except Exception as error:
                st.error(f'Не удалось получить результаты: {error}')
            else:
                st.subheader('Последние 10 фродовых транзакций')
                if fraud:
                    st.dataframe(pd.DataFrame(fraud), hide_index=True)
                else:
                    st.info('Фродовых транзакций пока нет')
                st.subheader(
                    f'Распределение скоров: последние {len(scores)} транзакций'
                )
                if scores:
                    counts, edges = np.histogram(scores, bins=10, range=(0, 1))
                    st.vega_lite_chart(
                        pd.DataFrame(
                            {
                                'start': edges[:-1],
                                'end': edges[1:],
                                'transactions': counts,
                            }
                        ),
                        {
                            'mark': {'type': 'bar', 'color': '#60A5FA'},
                            'encoding': {
                                'x': {
                                    'field': 'start',
                                    'type': 'quantitative',
                                    'bin': 'binned',
                                    'title': 'Скор',
                                    'scale': {'domain': [0, 1]},
                                    'axis': {'format': '.1f', 'tickCount': 11},
                                },
                                'x2': {'field': 'end'},
                                'y': {
                                    'field': 'transactions',
                                    'type': 'quantitative',
                                    'title': 'Транзакции',
                                    'axis': {'tickMinStep': 1},
                                },
                            },
                        },
                        use_container_width=True,
                    )
                else:
                    st.info('Результатов скоринга пока нет')


if __name__ == '__main__':
    main()
