-- Phase 4.5: recognition session and controlled camera-test logging.
-- First migration for these tables. It does not alter attendance/evaluation data.

create extension if not exists pgcrypto;

create table if not exists public.recognition_sessions (
    id uuid primary key default gen_random_uuid(),
    class_code text not null,
    subject_code text not null,
    session_type text not null,
    model_name text not null,
    model_threshold double precision not null,
    config_snapshot jsonb not null default '{}'::jsonb,
    started_at timestamptz not null default now(),
    ended_at timestamptz,
    notes text,
    created_at timestamptz not null default now(),

    constraint recognition_sessions_class_code_not_blank
        check (length(btrim(class_code)) > 0),
    constraint recognition_sessions_subject_code_not_blank
        check (length(btrim(subject_code)) > 0),
    constraint recognition_sessions_type_valid
        check (session_type in ('camera_test', 'real_class')),
    constraint recognition_sessions_model_name_not_blank
        check (length(btrim(model_name)) > 0),
    constraint recognition_sessions_threshold_valid
        check (model_threshold between -1.0 and 1.0),
    constraint recognition_sessions_config_is_object
        check (jsonb_typeof(config_snapshot) = 'object'),
    constraint recognition_sessions_end_after_start
        check (ended_at is null or ended_at >= started_at)
);

create table if not exists public.recognition_test_trials (
    id uuid primary key default gen_random_uuid(),
    session_id uuid not null,
    trial_number integer not null,
    expected_user_id text not null,
    distance_m double precision not null,
    detected boolean not null default false,
    detector_confidence double precision,
    predicted_user_id text,
    predicted_name text,
    similarity double precision,
    threshold double precision not null,
    recognized boolean not null default false,
    detection_time_ms double precision,
    recognition_time_ms double precision,
    notes text,
    created_at timestamptz not null default now(),

    constraint recognition_test_trials_session_fk
        foreign key (session_id)
        references public.recognition_sessions(id)
        on delete cascade,
    constraint recognition_test_trials_number_positive
        check (trial_number > 0),
    constraint recognition_test_trials_expected_user_not_blank
        check (length(btrim(expected_user_id)) > 0),
    constraint recognition_test_trials_distance_nonnegative
        check (distance_m >= 0),
    constraint recognition_test_trials_detector_confidence_valid
        check (
            detector_confidence is null
            or detector_confidence between 0.0 and 1.0
        ),
    constraint recognition_test_trials_similarity_valid
        check (similarity is null or similarity between -1.0 and 1.0),
    constraint recognition_test_trials_threshold_valid
        check (threshold between -1.0 and 1.0),
    constraint recognition_test_trials_detection_time_nonnegative
        check (detection_time_ms is null or detection_time_ms >= 0),
    constraint recognition_test_trials_recognition_time_nonnegative
        check (recognition_time_ms is null or recognition_time_ms >= 0),
    constraint recognition_test_trials_session_trial_unique
        unique (session_id, trial_number)
);

create index if not exists recognition_sessions_class_code_idx
    on public.recognition_sessions (class_code);

create index if not exists recognition_sessions_subject_code_idx
    on public.recognition_sessions (subject_code);

create index if not exists recognition_sessions_started_at_idx
    on public.recognition_sessions (started_at desc);

create index if not exists recognition_test_trials_session_id_idx
    on public.recognition_test_trials (session_id);

create index if not exists recognition_test_trials_expected_user_id_idx
    on public.recognition_test_trials (expected_user_id);

create index if not exists recognition_test_trials_created_at_idx
    on public.recognition_test_trials (created_at desc);

alter table public.recognition_sessions enable row level security;
alter table public.recognition_test_trials enable row level security;

-- No public/anonymous policies are created. A service-role client bypasses RLS.
-- If the application uses another role, add narrowly scoped authenticated
-- policies appropriate to that deployment; do not add an anonymous allow-all.
