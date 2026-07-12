BEGIN;

DROP SCHEMA IF EXISTS public CASCADE;
CREATE SCHEMA public;
COMMENT ON SCHEMA public IS 'Cashier monitoring and video analytics schema';

CREATE TABLE organizations (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    name text NOT NULL,
    code text NOT NULL UNIQUE,
    timezone text NOT NULL DEFAULT 'UTC',
    is_active boolean NOT NULL DEFAULT true,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE stores (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    organization_id bigint NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    name text NOT NULL,
    code text NOT NULL,
    city text,
    address text,
    is_active boolean NOT NULL DEFAULT true,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (organization_id, code)
);

CREATE TABLE workplaces (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    store_id bigint NOT NULL REFERENCES stores(id) ON DELETE CASCADE,
    name text NOT NULL,
    workplace_type text NOT NULL DEFAULT 'checkout',
    external_id text,
    is_active boolean NOT NULL DEFAULT true,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (store_id, external_id)
);

CREATE TABLE cameras (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    store_id bigint NOT NULL REFERENCES stores(id) ON DELETE CASCADE,
    workplace_id bigint REFERENCES workplaces(id) ON DELETE SET NULL,
    name text NOT NULL,
    code text NOT NULL,
    manufacturer text,
    model text,
    nvr_channel text,
    location_description text,
    status text NOT NULL DEFAULT 'unknown'
        CHECK (status IN ('unknown', 'online', 'offline', 'degraded', 'error')),
    last_online_at timestamptz,
    last_frame_at timestamptz,
    processing_enabled boolean NOT NULL DEFAULT true,
    is_active boolean NOT NULL DEFAULT true,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (store_id, code)
);

CREATE TABLE camera_streams (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    camera_id bigint NOT NULL REFERENCES cameras(id) ON DELETE CASCADE,
    stream_type text NOT NULL,
    stream_url text NOT NULL,
    subtype text,
    width integer CHECK (width IS NULL OR width > 0),
    height integer CHECK (height IS NULL OR height > 0),
    source_fps double precision CHECK (source_fps IS NULL OR source_fps >= 0),
    process_fps double precision CHECK (process_fps IS NULL OR process_fps >= 0),
    transport text NOT NULL DEFAULT 'tcp',
    is_primary boolean NOT NULL DEFAULT false,
    is_enabled boolean NOT NULL DEFAULT true,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE UNIQUE INDEX uq_camera_primary_stream
    ON camera_streams(camera_id, stream_type) WHERE is_primary;

CREATE TABLE roi_types (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    code text NOT NULL UNIQUE,
    name text NOT NULL
);

CREATE TABLE camera_rois (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    camera_id bigint NOT NULL REFERENCES cameras(id) ON DELETE CASCADE,
    roi_type_id bigint NOT NULL REFERENCES roi_types(id) ON DELETE RESTRICT,
    name text NOT NULL,
    shape_type text NOT NULL DEFAULT 'polygon'
        CHECK (shape_type IN ('rectangle', 'polygon', 'line', 'point')),
    coordinates jsonb NOT NULL,
    is_enabled boolean NOT NULL DEFAULT true,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (camera_id, name),
    CHECK (jsonb_typeof(coordinates) IN ('array', 'object'))
);

CREATE TABLE models (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    name text NOT NULL,
    model_type text NOT NULL,
    framework text NOT NULL,
    task_type text NOT NULL,
    description text,
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (name, model_type)
);

CREATE TABLE model_versions (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    model_id bigint NOT NULL REFERENCES models(id) ON DELETE CASCADE,
    version text NOT NULL,
    weights_path text NOT NULL,
    config jsonb NOT NULL DEFAULT '{}'::jsonb,
    confidence_threshold double precision
        CHECK (confidence_threshold IS NULL OR confidence_threshold BETWEEN 0 AND 1),
    iou_threshold double precision
        CHECK (iou_threshold IS NULL OR iou_threshold BETWEEN 0 AND 1),
    metrics jsonb NOT NULL DEFAULT '{}'::jsonb,
    is_active boolean NOT NULL DEFAULT true,
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (model_id, version)
);

CREATE TABLE camera_models (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    camera_id bigint NOT NULL REFERENCES cameras(id) ON DELETE CASCADE,
    model_version_id bigint NOT NULL REFERENCES model_versions(id) ON DELETE CASCADE,
    roi_id bigint REFERENCES camera_rois(id) ON DELETE SET NULL,
    process_fps double precision CHECK (process_fps IS NULL OR process_fps >= 0),
    confidence_threshold double precision
        CHECK (confidence_threshold IS NULL OR confidence_threshold BETWEEN 0 AND 1),
    config jsonb NOT NULL DEFAULT '{}'::jsonb,
    is_enabled boolean NOT NULL DEFAULT true,
    UNIQUE (camera_id, model_version_id, roi_id)
);

CREATE TABLE processing_sessions (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    camera_id bigint NOT NULL REFERENCES cameras(id) ON DELETE CASCADE,
    stream_id bigint REFERENCES camera_streams(id) ON DELETE SET NULL,
    worker_name text NOT NULL,
    worker_host text,
    started_at timestamptz NOT NULL DEFAULT now(),
    finished_at timestamptz,
    status text NOT NULL DEFAULT 'running'
        CHECK (status IN ('starting', 'running', 'stopped', 'failed')),
    frames_read bigint NOT NULL DEFAULT 0 CHECK (frames_read >= 0),
    frames_processed bigint NOT NULL DEFAULT 0 CHECK (frames_processed >= 0),
    frames_dropped bigint NOT NULL DEFAULT 0 CHECK (frames_dropped >= 0),
    average_fps double precision CHECK (average_fps IS NULL OR average_fps >= 0),
    average_latency_ms double precision CHECK (average_latency_ms IS NULL OR average_latency_ms >= 0),
    error_message text,
    created_at timestamptz NOT NULL DEFAULT now(),
    CHECK (finished_at IS NULL OR finished_at >= started_at)
);

CREATE TABLE employees (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    organization_id bigint NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    external_id text,
    full_name text NOT NULL,
    role text NOT NULL DEFAULT 'cashier',
    is_active boolean NOT NULL DEFAULT true,
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (organization_id, external_id)
);

CREATE TABLE shifts (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    store_id bigint NOT NULL REFERENCES stores(id) ON DELETE CASCADE,
    workplace_id bigint REFERENCES workplaces(id) ON DELETE SET NULL,
    employee_id bigint REFERENCES employees(id) ON DELETE SET NULL,
    external_shift_id text,
    opened_at timestamptz NOT NULL,
    closed_at timestamptz,
    status text NOT NULL DEFAULT 'open'
        CHECK (status IN ('planned', 'open', 'closed', 'cancelled')),
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (store_id, external_shift_id),
    CHECK (closed_at IS NULL OR closed_at >= opened_at)
);

CREATE TABLE event_types (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    code text NOT NULL UNIQUE,
    name text NOT NULL,
    category text NOT NULL,
    default_severity text NOT NULL DEFAULT 'info'
        CHECK (default_severity IN ('info', 'low', 'medium', 'high', 'critical')),
    description text
);

CREATE TABLE rules (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    code text NOT NULL UNIQUE,
    name text NOT NULL,
    event_type_id bigint NOT NULL REFERENCES event_types(id) ON DELETE RESTRICT,
    rule_type text NOT NULL,
    severity text NOT NULL DEFAULT 'medium'
        CHECK (severity IN ('info', 'low', 'medium', 'high', 'critical')),
    conditions jsonb NOT NULL DEFAULT '{}'::jsonb,
    settings jsonb NOT NULL DEFAULT '{}'::jsonb,
    is_active boolean NOT NULL DEFAULT true,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE rule_assignments (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    rule_id bigint NOT NULL REFERENCES rules(id) ON DELETE CASCADE,
    organization_id bigint REFERENCES organizations(id) ON DELETE CASCADE,
    store_id bigint REFERENCES stores(id) ON DELETE CASCADE,
    workplace_id bigint REFERENCES workplaces(id) ON DELETE CASCADE,
    camera_id bigint REFERENCES cameras(id) ON DELETE CASCADE,
    settings_override jsonb NOT NULL DEFAULT '{}'::jsonb,
    is_enabled boolean NOT NULL DEFAULT true,
    created_at timestamptz NOT NULL DEFAULT now(),
    CHECK (num_nonnulls(organization_id, store_id, workplace_id, camera_id) >= 1)
);

CREATE TABLE detections (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    camera_id bigint NOT NULL REFERENCES cameras(id) ON DELETE CASCADE,
    processing_session_id bigint REFERENCES processing_sessions(id) ON DELETE SET NULL,
    model_version_id bigint REFERENCES model_versions(id) ON DELETE SET NULL,
    roi_id bigint REFERENCES camera_rois(id) ON DELETE SET NULL,
    detected_at timestamptz NOT NULL,
    frame_number bigint CHECK (frame_number IS NULL OR frame_number >= 0),
    class_name text NOT NULL,
    confidence double precision NOT NULL CHECK (confidence BETWEEN 0 AND 1),
    x1 double precision NOT NULL,
    y1 double precision NOT NULL,
    x2 double precision NOT NULL,
    y2 double precision NOT NULL,
    track_id text,
    attributes jsonb NOT NULL DEFAULT '{}'::jsonb,
    CHECK (x2 >= x1 AND y2 >= y1)
);

CREATE TABLE tracks (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    camera_id bigint NOT NULL REFERENCES cameras(id) ON DELETE CASCADE,
    processing_session_id bigint REFERENCES processing_sessions(id) ON DELETE SET NULL,
    tracker_track_id text NOT NULL,
    object_type text NOT NULL,
    first_seen_at timestamptz NOT NULL,
    last_seen_at timestamptz NOT NULL,
    max_confidence double precision CHECK (max_confidence IS NULL OR max_confidence BETWEEN 0 AND 1),
    duration_ms bigint CHECK (duration_ms IS NULL OR duration_ms >= 0),
    start_roi_id bigint REFERENCES camera_rois(id) ON DELETE SET NULL,
    end_roi_id bigint REFERENCES camera_rois(id) ON DELETE SET NULL,
    attributes jsonb NOT NULL DEFAULT '{}'::jsonb,
    UNIQUE (processing_session_id, tracker_track_id),
    CHECK (last_seen_at >= first_seen_at)
);

CREATE TABLE analytics_events (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    organization_id bigint NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    store_id bigint NOT NULL REFERENCES stores(id) ON DELETE CASCADE,
    workplace_id bigint REFERENCES workplaces(id) ON DELETE SET NULL,
    camera_id bigint REFERENCES cameras(id) ON DELETE SET NULL,
    event_type_id bigint NOT NULL REFERENCES event_types(id) ON DELETE RESTRICT,
    rule_id bigint REFERENCES rules(id) ON DELETE SET NULL,
    started_at timestamptz NOT NULL,
    finished_at timestamptz,
    severity text NOT NULL DEFAULT 'medium'
        CHECK (severity IN ('info', 'low', 'medium', 'high', 'critical')),
    status text NOT NULL DEFAULT 'open'
        CHECK (status IN ('open', 'acknowledged', 'confirmed', 'dismissed', 'resolved')),
    confidence double precision CHECK (confidence IS NULL OR confidence BETWEEN 0 AND 1),
    duration_ms bigint CHECK (duration_ms IS NULL OR duration_ms >= 0),
    employee_id bigint REFERENCES employees(id) ON DELETE SET NULL,
    shift_id bigint REFERENCES shifts(id) ON DELETE SET NULL,
    external_order_id text,
    external_receipt_id text,
    title text NOT NULL,
    description text,
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    CHECK (finished_at IS NULL OR finished_at >= started_at)
);

CREATE TABLE event_objects (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    event_id bigint NOT NULL REFERENCES analytics_events(id) ON DELETE CASCADE,
    object_role text,
    object_type text NOT NULL,
    track_id bigint REFERENCES tracks(id) ON DELETE SET NULL,
    confidence double precision CHECK (confidence IS NULL OR confidence BETWEEN 0 AND 1),
    bbox jsonb,
    attributes jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE event_reviews (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    event_id bigint NOT NULL REFERENCES analytics_events(id) ON DELETE CASCADE,
    reviewer_id bigint REFERENCES employees(id) ON DELETE SET NULL,
    decision text NOT NULL,
    comment text,
    previous_status text,
    new_status text,
    reviewed_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE notifications (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    event_id bigint NOT NULL REFERENCES analytics_events(id) ON DELETE CASCADE,
    channel text NOT NULL,
    recipient text NOT NULL,
    status text NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending', 'sending', 'sent', 'failed', 'acknowledged')),
    payload jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    sent_at timestamptz,
    acknowledged_at timestamptz,
    error_message text
);

CREATE TABLE event_evidence (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    event_id bigint NOT NULL REFERENCES analytics_events(id) ON DELETE CASCADE,
    evidence_type text NOT NULL,
    storage_type text NOT NULL,
    file_path text NOT NULL,
    mime_type text,
    file_size bigint CHECK (file_size IS NULL OR file_size >= 0),
    captured_at timestamptz NOT NULL,
    expires_at timestamptz,
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    CHECK (expires_at IS NULL OR expires_at >= captured_at)
);

CREATE TABLE external_events (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    source_system text NOT NULL,
    event_type text NOT NULL,
    external_event_id text NOT NULL,
    store_id bigint REFERENCES stores(id) ON DELETE SET NULL,
    workplace_id bigint REFERENCES workplaces(id) ON DELETE SET NULL,
    occurred_at timestamptz NOT NULL,
    received_at timestamptz NOT NULL DEFAULT now(),
    payload jsonb NOT NULL,
    processing_status text NOT NULL DEFAULT 'pending'
        CHECK (processing_status IN ('pending', 'processing', 'processed', 'failed', 'ignored')),
    processing_error text,
    UNIQUE (source_system, external_event_id)
);

CREATE TABLE camera_metrics (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    camera_id bigint NOT NULL REFERENCES cameras(id) ON DELETE CASCADE,
    recorded_at timestamptz NOT NULL DEFAULT now(),
    status text NOT NULL DEFAULT 'unknown',
    input_fps double precision CHECK (input_fps IS NULL OR input_fps >= 0),
    processing_fps double precision CHECK (processing_fps IS NULL OR processing_fps >= 0),
    latency_ms double precision CHECK (latency_ms IS NULL OR latency_ms >= 0),
    dropped_frames bigint CHECK (dropped_frames IS NULL OR dropped_frames >= 0),
    reconnect_count bigint CHECK (reconnect_count IS NULL OR reconnect_count >= 0),
    cpu_percent double precision CHECK (cpu_percent IS NULL OR cpu_percent BETWEEN 0 AND 100),
    gpu_percent double precision CHECK (gpu_percent IS NULL OR gpu_percent BETWEEN 0 AND 100),
    gpu_memory_mb double precision CHECK (gpu_memory_mb IS NULL OR gpu_memory_mb >= 0),
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX idx_stores_organization ON stores(organization_id);
CREATE INDEX idx_workplaces_store ON workplaces(store_id);
CREATE INDEX idx_cameras_store ON cameras(store_id);
CREATE INDEX idx_cameras_workplace ON cameras(workplace_id);
CREATE INDEX idx_camera_streams_camera ON camera_streams(camera_id);
CREATE INDEX idx_camera_rois_camera ON camera_rois(camera_id);
CREATE INDEX idx_camera_models_camera ON camera_models(camera_id);
CREATE INDEX idx_processing_sessions_camera_started ON processing_sessions(camera_id, started_at DESC);
CREATE INDEX idx_detections_camera_time ON detections(camera_id, detected_at DESC);
CREATE INDEX idx_detections_session_frame ON detections(processing_session_id, frame_number);
CREATE INDEX idx_detections_track ON detections(camera_id, track_id) WHERE track_id IS NOT NULL;
CREATE INDEX idx_tracks_camera_time ON tracks(camera_id, first_seen_at DESC);
CREATE INDEX idx_shifts_employee_time ON shifts(employee_id, opened_at DESC);
CREATE INDEX idx_analytics_events_org_time ON analytics_events(organization_id, started_at DESC);
CREATE INDEX idx_analytics_events_store_time ON analytics_events(store_id, started_at DESC);
CREATE INDEX idx_analytics_events_camera_time ON analytics_events(camera_id, started_at DESC);
CREATE INDEX idx_analytics_events_status ON analytics_events(status, severity, started_at DESC);
CREATE INDEX idx_event_objects_event ON event_objects(event_id);
CREATE INDEX idx_event_reviews_event ON event_reviews(event_id, reviewed_at DESC);
CREATE INDEX idx_notifications_status ON notifications(status, created_at);
CREATE INDEX idx_event_evidence_event ON event_evidence(event_id, captured_at);
CREATE INDEX idx_external_events_status ON external_events(processing_status, received_at);
CREATE INDEX idx_camera_metrics_camera_time ON camera_metrics(camera_id, recorded_at DESC);
CREATE INDEX idx_detections_attributes_gin ON detections USING gin(attributes);
CREATE INDEX idx_analytics_events_metadata_gin ON analytics_events USING gin(metadata);
CREATE INDEX idx_external_events_payload_gin ON external_events USING gin(payload);

CREATE OR REPLACE FUNCTION set_updated_at()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$;

CREATE TRIGGER organizations_set_updated_at BEFORE UPDATE ON organizations
FOR EACH ROW EXECUTE FUNCTION set_updated_at();
CREATE TRIGGER stores_set_updated_at BEFORE UPDATE ON stores
FOR EACH ROW EXECUTE FUNCTION set_updated_at();
CREATE TRIGGER workplaces_set_updated_at BEFORE UPDATE ON workplaces
FOR EACH ROW EXECUTE FUNCTION set_updated_at();
CREATE TRIGGER cameras_set_updated_at BEFORE UPDATE ON cameras
FOR EACH ROW EXECUTE FUNCTION set_updated_at();
CREATE TRIGGER camera_rois_set_updated_at BEFORE UPDATE ON camera_rois
FOR EACH ROW EXECUTE FUNCTION set_updated_at();
CREATE TRIGGER rules_set_updated_at BEFORE UPDATE ON rules
FOR EACH ROW EXECUTE FUNCTION set_updated_at();
CREATE TRIGGER analytics_events_set_updated_at BEFORE UPDATE ON analytics_events
FOR EACH ROW EXECUTE FUNCTION set_updated_at();

INSERT INTO roi_types (code, name) VALUES
    ('scan_zone', 'Зона сканирования'),
    ('customer_zone', 'Зона клиента'),
    ('cashier_zone', 'Зона кассира'),
    ('bag_zone', 'Зона упаковки'),
    ('cash_drawer_zone', 'Зона денежного ящика'),
    ('exit_zone', 'Зона выхода');

INSERT INTO event_types (code, name, category, default_severity, description) VALUES
    ('customer_present', 'Клиент подошёл', 'service', 'info', 'Клиент появился в зоне обслуживания'),
    ('customer_left', 'Клиент ушёл', 'service', 'info', 'Клиент покинул зону обслуживания'),
    ('cashier_present', 'Кассир присутствует', 'staff', 'info', 'Кассир появился на рабочем месте'),
    ('cashier_left', 'Кассир отсутствует', 'staff', 'medium', 'Кассир покинул рабочее место'),
    ('no_cashier', 'Нет кассира', 'staff', 'high', 'Клиент ожидает при отсутствии кассира'),
    ('object_detected', 'Объект обнаружен', 'object', 'info', 'Объект обнаружен в контролируемой зоне'),
    ('product_counted', 'Товар посчитан', 'object', 'info', 'Новый товар добавлен в счётчик'),
    ('hand_raised', 'Рука поднята', 'pose', 'low', 'Обнаружено поднятое положение руки'),
    ('processing_error', 'Ошибка обработки', 'system', 'high', 'Ошибка видеопроцессинга или аналитики'),
    ('camera_offline', 'Камера недоступна', 'system', 'high', 'Камера или поток недоступны');

COMMIT;

