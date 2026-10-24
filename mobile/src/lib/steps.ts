import type { Job, JobStatus } from '@/api/types';

export type StepState = 'done' | 'active' | 'waiting' | 'pending' | 'failed';

export interface Step {
  key: string;
  label: string;
  state: StepState;
}

const LABELS = [
  ['read', 'Reading your notes'],
  ['search', 'Finding web sources'],
  ['approve', 'You approve the sources'],
  ['write', 'Writing flashcards'],
  ['check', 'Checking quality'],
] as const;

export type Tone = 'neutral' | 'success' | 'danger' | 'warning' | 'info';

export const JOB_STATUS_LABEL: Record<JobStatus, { label: string; tone: Tone }> = {
  queued: { label: 'Waiting to start', tone: 'neutral' },
  researching: { label: 'Reading & searching', tone: 'info' },
  awaiting_approval: { label: 'Needs your approval', tone: 'warning' },
  approved: { label: 'Waiting to write', tone: 'info' },
  generating: { label: 'Writing cards', tone: 'info' },
  completed: { label: 'Ready', tone: 'success' },
  failed: { label: 'Failed', tone: 'danger' },
};

/** Statuses where the server is working on its own, so the client should keep polling. */
export const ACTIVE_STATUSES: readonly JobStatus[] = ['queued', 'researching', 'approved', 'generating'];

export function isActive(status: JobStatus): boolean {
  return ACTIVE_STATUSES.includes(status);
}

/** Index of the step a job is currently on (5 means everything is finished). */
export function currentStepIndex(job: Pick<Job, 'status' | 'stage' | 'auto_approve'>): number {
  switch (job.status) {
    case 'queued':
      return 0;
    case 'researching':
      return job.stage === 'enrich' ? 1 : 0;
    case 'awaiting_approval':
      return 2;
    case 'approved':
      return 3;
    case 'generating':
      return job.stage === 'verify' ? 4 : 3;
    case 'completed':
      return 5;
    case 'failed':
      // The server keeps `stage` on failure, so the step that broke can be shown.
      return STAGE_STEP[job.stage ?? ''] ?? 0;
  }
}

const STAGE_STEP: Record<string, number> = { understand: 0, enrich: 1, generate: 3, verify: 4 };

/** Stepper for the progress screen. */
export function stepsFor(job: Pick<Job, 'status' | 'stage' | 'auto_approve'>): Step[] {
  const current = currentStepIndex(job);
  // When approval is skipped (auto-approve, or no web sources found) the job simply moves past that step.
  return LABELS.map(([key, label], i) => {
    let state: StepState;
    if (job.status === 'failed' && i === current) state = 'failed';
    else if (i < current) state = 'done';
    else if (i === current) state = job.status === 'awaiting_approval' ? 'waiting' : 'active';
    else state = 'pending';
    return { key, label, state };
  });
}
