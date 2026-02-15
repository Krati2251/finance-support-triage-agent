-- ============================================
-- Finance Support Triage Agent — Database Schema
-- ============================================

-- Enum types for constrained columns
CREATE TYPE ticket_status   AS ENUM ('New', 'In Progress', 'Resolved', 'Closed');
CREATE TYPE ticket_priority AS ENUM ('High', 'Medium', 'Low');
CREATE TYPE ticket_category AS ENUM ('Fraud', 'Payment Issue', 'General');

-- Tickets table
CREATE TABLE IF NOT EXISTS tickets (
    id            UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    customer_name VARCHAR(255)    NOT NULL,
    email_body    TEXT            NOT NULL,
    status        ticket_status   NOT NULL DEFAULT 'New',
    priority      ticket_priority NOT NULL DEFAULT 'Medium',
    category      ticket_category NOT NULL DEFAULT 'General',
    sentiment     VARCHAR(50),
    intent        VARCHAR(500),
    summary       TEXT,
    transaction_id VARCHAR(100),
    amount        VARCHAR(50),
    draft_response TEXT,
    created_at    TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

-- Index for common query patterns
CREATE INDEX IF NOT EXISTS idx_tickets_status     ON tickets (status);
CREATE INDEX IF NOT EXISTS idx_tickets_priority   ON tickets (priority);
CREATE INDEX IF NOT EXISTS idx_tickets_category   ON tickets (category);
CREATE INDEX IF NOT EXISTS idx_tickets_created_at ON tickets (created_at DESC);
