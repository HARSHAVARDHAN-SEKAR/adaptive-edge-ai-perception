#!/bin/sh
# edge-perception container entrypoint
#   dashboard  -> live telemetry UI on :8090 (default)
#   demo       -> 300-frame adaptive mission in the console
#   test       -> run the test suite
#   bench      -> Experiments 1 & 2
#   viewer     -> record demo video to /app/assets/demo_mission.mp4
# anything else is executed verbatim (e.g. bash, python ...)
set -e
case "$1" in
  dashboard) exec python dashboard/server.py --source synthetic ;;
  demo)      exec python run_pipeline.py --source synthetic --frames 300 ;;
  test)      exec python tests/test_pipeline.py ;;
  bench)     python -m benchmark.latency_test
             exec python -m benchmark.fps_test --backend auto ;;
  viewer)    exec python viewer.py --source synthetic --frames 300 \
                  --record assets/demo_mission.mp4 ;;
  real-dashboard)
             exec python dashboard/server.py \
                  --source "${SOURCE:-synthetic}" \
                  --backend real --device "${DEVICE:-cpu}" ;;
  *)         exec "$@" ;;
esac
