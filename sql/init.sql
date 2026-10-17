create table scores (
    id bigint generated always as identity primary key,
    transaction_id uuid not null unique,
    score double precision not null check (score >= 0 and score <= 1),
    fraud_flag smallint not null check (fraud_flag in (0, 1)),
    model_version text not null,
    created_at timestamptz not null default current_timestamp
);

create index scores_fraud_latest on scores (id desc) where fraud_flag = 1;

create table labeled_transactions (
    data_month date not null,
    source_row bigint not null,
    transaction_time timestamp not null,
    merch text,
    cat_id text,
    amount double precision,
    name_1 text,
    name_2 text,
    gender text,
    street text,
    one_city text,
    us_state text,
    post_code text,
    lat double precision,
    lon double precision,
    population_city double precision,
    jobs text,
    merchant_lat double precision,
    merchant_lon double precision,
    target smallint,
    primary key (data_month, source_row)
);

create table training_state (
    id integer primary key check (id = 1),
    next_month date not null
);

create table run_months (
    flow text not null,
    run_id text not null,
    month date not null,
    primary key (flow, run_id)
);

create table training_cycles (
    month date primary key,
    version text not null,
    published boolean not null,
    decision jsonb not null
);

create table supplied_months (
    month date primary key,
    rows bigint not null
);
