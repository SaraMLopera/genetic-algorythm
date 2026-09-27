from flask import Flask, render_template, request, jsonify, Response
from genetic_algorithm import GeneticAlgorithmNQueens
import json
import threading
import time

app = Flask(__name__)

# Global GA instance and its lock
_ga: GeneticAlgorithmNQueens | None = None
_ga_lock = threading.Lock()
_stream_active = False


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/start", methods=["POST"])
def start():
    """Initialise a new GA run with the given parameters."""
    global _ga, _stream_active
    data = request.get_json(force=True)

    n               = int(data.get("n", 8))
    population_size = int(data.get("population_size", 100))
    mutation_rate   = float(data.get("mutation_rate", 0.05))
    elite_size      = int(data.get("elite_size", 10))
    max_generations = int(data.get("max_generations", 1000))
    tournament_size = int(data.get("tournament_size", 5))

    # Clamp / validate
    n               = max(4, min(n, 20))
    population_size = max(20, min(population_size, 500))
    mutation_rate   = max(0.0, min(mutation_rate, 1.0))
    elite_size      = max(1, min(elite_size, population_size // 2))
    max_generations = max(10, min(max_generations, 5000))
    tournament_size = max(2, min(tournament_size, population_size))

    with _ga_lock:
        _stream_active = False          # stop any running stream
        _ga = GeneticAlgorithmNQueens(
            n=n,
            population_size=population_size,
            mutation_rate=mutation_rate,
            elite_size=elite_size,
            max_generations=max_generations,
            tournament_size=tournament_size,
        )
        _ga.initialize_population()

    return jsonify({"status": "ok", "max_fitness": _ga.max_fitness, "n": n})


@app.route("/api/step", methods=["POST"])
def step():
    """Advance the GA by one generation and return stats."""
    with _ga_lock:
        if _ga is None:
            return jsonify({"error": "GA not initialised"}), 400
        if _ga.solved or _ga.generation >= _ga.max_generations:
            return jsonify({**_ga.get_state(), "done": True})
        stats = _ga.evolve_one_generation()
        state = _ga.get_state()
        state["done"] = _ga.solved or _ga.generation >= _ga.max_generations
    return jsonify(state)


@app.route("/api/stream")
def stream():
    """
    Server-Sent Events endpoint that streams one generation per tick
    until the GA is solved or exhausted.
    """
    global _stream_active

    delay = float(request.args.get("delay", 0.05))   # seconds between gens

    def generate():
        global _stream_active
        _stream_active = True
        try:
            while True:
                with _ga_lock:
                    if _ga is None:
                        yield "data: {\"error\": \"GA not initialised\"}\n\n"
                        return
                    if not _stream_active:
                        return
                    if _ga.solved or _ga.generation >= _ga.max_generations:
                        state = _ga.get_state()
                        state["done"] = True
                        yield f"data: {json.dumps(state)}\n\n"
                        return
                    _ga.evolve_one_generation()
                    state = _ga.get_state()
                    state["done"] = False

                yield f"data: {json.dumps(state)}\n\n"
                time.sleep(delay)
        finally:
            _stream_active = False

    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.route("/api/stop", methods=["POST"])
def stop():
    global _stream_active
    _stream_active = False
    return jsonify({"status": "stopped"})


@app.route("/api/state")
def state():
    with _ga_lock:
        if _ga is None:
            return jsonify({"error": "GA not initialised"}), 400
        return jsonify(_ga.get_state())


if __name__ == "__main__":
    app.run(debug=True, threaded=True)
