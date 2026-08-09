let benchmarkRunning = false;


function getDatasetRoot() {
    return document
        .getElementById("datasetRoot")
        .value
        .trim();
}


function setBenchmarkState(isRunning) {
    benchmarkRunning = isRunning;

    document.getElementById("runBuffaloBtn").disabled =
        isRunning;

    document.getElementById("runFaceNetBtn").disabled =
        isRunning;

    document.getElementById("runAllBtn").disabled =
        isRunning;
}


async function runBenchmark(modelName) {
    if (benchmarkRunning) {
        return;
    }

    const datasetRoot = getDatasetRoot();

    if (!datasetRoot) {
        alert("Please enter the dataset root.");
        return;
    }

    const status =
        document.getElementById("evaluationStatus");

    setBenchmarkState(true);

    status.innerText =
        "Running benchmark for " +
        modelName +
        ". Please wait...";

    try {
        const response = await fetch("/run-benchmark", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                model: modelName,
                dataset_root: datasetRoot
            })
        });

        const result = await response.json();

        if (result.success) {
            const accuracy =
                formatPercent(
                    result.metrics?.accuracy
                );

            status.innerText =
                modelName +
                " benchmark completed. Accuracy: " +
                accuracy;

            await loadComparison();
        } else {
            status.innerText =
                "Benchmark failed: " +
                (result.message || "Unknown error");
        }
    } catch (error) {
        status.innerText =
            "Benchmark request failed: " +
            error.message;
    } finally {
        setBenchmarkState(false);
    }
}


async function runAllBenchmarks() {
    if (benchmarkRunning) {
        return;
    }

    const datasetRoot = getDatasetRoot();

    if (!datasetRoot) {
        alert("Please enter the dataset root.");
        return;
    }

    const confirmed = confirm(
        "Run all supported models? This may take several minutes."
    );

    if (!confirmed) {
        return;
    }

    const status =
        document.getElementById("evaluationStatus");

    setBenchmarkState(true);

    status.innerText =
        "Running all model benchmarks. Please wait...";

    try {
        const response = await fetch(
            "/run-all-benchmarks",
            {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({
                    dataset_root: datasetRoot
                })
            }
        );

        const result = await response.json();

        if (result.success) {
            status.innerText =
                "All benchmarks completed successfully.";
        } else {
            status.innerText =
                "Benchmark finished with failures. " +
                "Successful: " +
                result.successful_models +
                ", Failed: " +
                result.failed_models;
        }

        await loadComparison();
    } catch (error) {
        status.innerText =
            "Run-all request failed: " +
            error.message;
    } finally {
        setBenchmarkState(false);
    }
}


async function loadComparison() {
    const tbody =
        document.getElementById(
            "evaluationTableBody"
        );

    tbody.innerHTML = `
        <tr>
            <td colspan="12">
                Loading evaluation results...
            </td>
        </tr>
    `;

    try {
        const response = await fetch(
            "/evaluation-latest-comparison"
        );

        const result = await response.json();

        tbody.innerHTML = "";

        if (
            !result.success ||
            !result.models ||
            result.models.length === 0
        ) {
            tbody.innerHTML = `
                <tr>
                    <td colspan="12">
                        No completed benchmark results found.
                    </td>
                </tr>
            `;
            return;
        }

        result.models.forEach(run => {
            const row =
                document.createElement("tr");

            row.innerHTML = `
                <td>${escapeHtml(run.model_name)}</td>

                <td>${formatPercent(
                    run.accuracy
                )}</td>

                <td>${formatPercent(
                    run.precision
                )}</td>

                <td>${formatPercent(
                    run.recall
                )}</td>

                <td>${formatPercent(
                    run.f1_score
                )}</td>

                <td>${formatPercent(
                    run.known_identity_accuracy
                )}</td>

                <td>${formatPercent(
                    run.false_accept_rate
                )}</td>

                <td>${formatPercent(
                    run.false_reject_rate
                )}</td>

                <td>${formatPercent(
                    run.wrong_identity_rate
                )}</td>

                <td>${formatMilliseconds(
                    run.average_processing_time_ms
                )}</td>

                <td>
                    ${Number(
                        run.correct_predictions || 0
                    )}
                    /
                    ${Number(
                        run.total_images || 0
                    )}
                </td>

                <td>${escapeHtml(
                    run.status || "unknown"
                )}</td>
            `;

            tbody.appendChild(row);
        });
    } catch (error) {
        tbody.innerHTML = `
            <tr>
                <td colspan="12">
                    Failed to load comparison:
                    ${escapeHtml(error.message)}
                </td>
            </tr>
        `;
    }
}


function formatPercent(value) {
    const numericValue = Number(value);

    if (!Number.isFinite(numericValue)) {
        return "0.0%";
    }

    return (
        numericValue * 100
    ).toFixed(1) + "%";
}


function formatMilliseconds(value) {
    const numericValue = Number(value);

    if (!Number.isFinite(numericValue)) {
        return "0.0 ms";
    }

    return numericValue.toFixed(1) + " ms";
}


function escapeHtml(value) {
    return String(value ?? "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
}


window.addEventListener("load", () => {
    loadComparison();
});