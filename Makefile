.PHONY: demo dashboard test bench video docker-up docker-test report
demo:       ; python3 run_pipeline.py --source synthetic --frames 300
dashboard:  ; python3 dashboard/server.py
test:       ; python3 tests/test_pipeline.py
bench:      ; python3 -m benchmark.latency_test && python3 -m benchmark.fps_test
video:      ; python3 viewer.py --source synthetic --frames 300 --record assets/demo_mission.mp4
report:     ; python3 docs/make_report_pdf.py
docker-up:  ; docker compose up dashboard
docker-test:; docker compose run --rm test
