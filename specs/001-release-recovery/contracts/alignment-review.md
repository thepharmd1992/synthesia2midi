# Contract: Guide Alignment Review

## Accepted outcomes

- Accepting Manual Fit sets current-session alignment review complete.
- Accepting Auto-Detect Tuning sets current-session alignment review complete
  before any optional assisted pressed-key scan begins.
- The control panel refreshes after the accepted outcome so the Guide advances
  immediately.

## Rejected or incomplete outcomes

- Opening an editor does not complete review.
- Canceling or rejecting either editor does not complete review.
- Failure to open an editor does not complete review.
- Canceling, retrying, keeping existing values, or finding no result in the
  subsequent assisted scan does not undo an already accepted alignment review.

## Invalidation

- Loading a new video clears current-session review before associated config is
  considered.
- Generating a replacement overlay set clears current-session review.
- Clearing calibration overlays clears current-session review.

## Compatibility

- Existing downstream no-key/exemplar calibration remains sufficient Guide
  evidence, including for projects created before this field existed.
- The review flag is never serialized to existing INI/JSON formats.
