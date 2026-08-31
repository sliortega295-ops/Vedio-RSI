# Executable cache search plan

1. Freeze and verify the live contract: clean start SHA, immutable baseline,
   exact environment, persistent storage, active UUID lease, and foreign GPU
   ownership. **Complete.**
2. Add an OFF-safe SANA cache controller and CLI/wrapper plumbing for exactly
   EasyCache, TeaCache, and TaylorSeer; add CPU tests for reset, family decisions,
   hit caps, Taylor forecast, and OFF identity. **In progress.**
3. Run static tests and create the pre-round infrastructure commit.
4. Explore EasyCache with real full runs, preserving each config and result in
   the trajectory.  Start conservatively, then tune only from measured hit/time/
   visual evidence.
5. Explore TeaCache at overlapping hit/time targets, using the SANA timestep-
   modulated signal and bounded residual replay.
6. Explore TaylorSeer at overlapping targets, first-order before second-order,
   and preserve the exact forecast history/damping/hit trace.
7. Use remaining rounds to fill meaningful matched-time gaps or verify close/
   noisy points.  Stop only at genuine convergence or the 20-round cap.
8. Validate the complete ledger, produce honest `DELIVERY.json`, commit the final
   small evidence, and notify the master for independent integration checks.
