
document.addEventListener("DOMContentLoaded", () => {
    "use strict";

    // ==========================================
    // 1. ELEMENTS
    // ==========================================

    const urlInput = document.getElementById("urlInput");
    const scanButton = document.getElementById("scanButton");
    const errorMessage = document.getElementById("errorMessage");
    const loadingCard = document.getElementById("loadingCard");

    const resultModal = document.getElementById("resultModal");
    const modalBackdrop = document.getElementById("modalBackdrop");
    const closeModal = document.getElementById("closeModal");
    const newScanButton = document.getElementById("newScanButton");

    const modalIcon = document.getElementById("modalIcon");
    const modalStatus = document.getElementById("modalStatus");
    const modalTitle = document.getElementById("modalTitle");
    const modalDescription = document.getElementById("modalDescription");
    const checkedUrl = document.getElementById("checkedUrl");

    // Website preview
    const previewStatus = document.getElementById("previewStatus");
    const previewDomain = document.getElementById("previewDomain");

    const websitePreviewPlaceholder =
        document.getElementById("websitePreviewPlaceholder");

    const websiteScreenshot =
        document.getElementById("websiteScreenshot");

    const previewPageTitle =
        document.getElementById("previewPageTitle");

    const previewReachability =
        document.getElementById("previewReachability");

    // Random Forest
    const randomForestCard =
        document.getElementById("randomForestCard");

    const randomForestResult =
        document.getElementById("randomForestResult");

    const randomForestConfidence =
        document.getElementById("randomForestConfidence");

    // Logistic Regression
    const logisticCard =
        document.getElementById("logisticCard");

    const logisticResult =
        document.getElementById("logisticResult");

    const logisticConfidence =
        document.getElementById("logisticConfidence");

    // Multinomial Naive Bayes
    const naiveBayesCard =
        document.getElementById("naiveBayesCard");

    const naiveBayesResult =
        document.getElementById("naiveBayesResult");

    const naiveBayesConfidence =
        document.getElementById("naiveBayesConfidence");


    // ==========================================
    // 2. INITIAL STATE
    // ==========================================

    if (loadingCard) {
        loadingCard.style.display = "none";
    }

    if (resultModal) {
        resultModal.style.display = "none";
        resultModal.setAttribute("aria-hidden", "true");
    }

    if (errorMessage) {
        errorMessage.style.display = "none";
    }

    resetModelResults();
    resetWebsitePreview();


    // ==========================================
    // 3. URL NORMALIZATION AND VALIDATION
    // ==========================================

    function normalizeUrl(value) {
        let url = value.trim();

        if (!/^[a-zA-Z][a-zA-Z0-9+.-]*:\/\//.test(url)) {
            url = "https://" + url;
        }

        return url;
    }

    function isValidUrl(value) {
        try {
            const parsed = new URL(value);

            if (!["http:", "https:"].includes(parsed.protocol)) {
                return false;
            }

            if (!parsed.hostname || /\s/.test(value)) {
                return false;
            }

            if (parsed.username || parsed.password) {
                return false;
            }

            const hostname = parsed.hostname;

            if (
                hostname !== "localhost" &&
                !hostname.includes(".") &&
                !hostname.startsWith("[")
            ) {
                return false;
            }

            return true;

        } catch {
            return false;
        }
    }


    // ==========================================
    // 4. ERROR HANDLING
    // ==========================================

    function showError(message) {
        if (!errorMessage) return;

        errorMessage.textContent = message;
        errorMessage.style.display = "block";

        errorMessage.setAttribute("role", "alert");
        errorMessage.setAttribute("aria-live", "assertive");
    }

    function hideError() {
        if (!errorMessage) return;

        errorMessage.textContent = "";
        errorMessage.style.display = "none";
    }


    // ==========================================
    // 5. LOADING STATE
    // ==========================================

    function showLoading() {
        if (loadingCard) {
            loadingCard.style.display = "flex";
            loadingCard.setAttribute("aria-live", "polite");
        }

        if (scanButton) {
            scanButton.disabled = true;
            scanButton.setAttribute("aria-busy", "true");
        }

        if (urlInput) {
            urlInput.disabled = true;
        }
    }

    function hideLoading() {
        if (loadingCard) {
            loadingCard.style.display = "none";
        }

        if (scanButton) {
            scanButton.disabled = false;
            scanButton.removeAttribute("aria-busy");
        }

        if (urlInput) {
            urlInput.disabled = false;
        }
    }


    // ==========================================
    // 6. MODEL RESULT DISPLAY
    // ==========================================

    function displayModelResult(
        card,
        resultElement,
        confidenceElement,
        result
    ) {
        if (!card || !resultElement || !confidenceElement) {
            return;
        }

        card.classList.remove(
            "safe",
            "phishing",
            "unavailable"
        );

        if (!result || !result.label) {
            resultElement.textContent = "Unavailable";

            confidenceElement.textContent =
                "No prediction returned";

            card.classList.add("unavailable");
            return;
        }

        const label = String(result.label).trim();
        const normalizedLabel = label.toLowerCase();

        resultElement.textContent = label;

        if (normalizedLabel === "legitimate") {
            card.classList.add("safe");

            const probability =
                Number(result.legitimate_probability);

            confidenceElement.textContent =
                Number.isFinite(probability)
                    ? `Legitimate score: ${probability.toFixed(2)}%`
                    : "Score unavailable";

        } else if (normalizedLabel === "phishing") {
            card.classList.add("phishing");

            const probability =
                Number(result.phishing_probability);

            confidenceElement.textContent =
                Number.isFinite(probability)
                    ? `Phishing score: ${probability.toFixed(2)}%`
                    : "Score unavailable";

        } else {
            card.classList.add("unavailable");
            confidenceElement.textContent =
                "Unrecognized model label";
        }
    }


    // ==========================================
    // 7. RESET MODEL RESULTS
    // ==========================================

    function resetModelResults() {
        const cards = [
            randomForestCard,
            logisticCard,
            naiveBayesCard
        ];

        const results = [
            randomForestResult,
            logisticResult,
            naiveBayesResult
        ];

        const confidences = [
            randomForestConfidence,
            logisticConfidence,
            naiveBayesConfidence
        ];

        cards.forEach(card => {
            if (card) {
                card.classList.remove(
                    "safe",
                    "phishing",
                    "unavailable"
                );
            }
        });

        results.forEach(element => {
            if (element) {
                element.textContent = "--";
            }
        });

        confidences.forEach(element => {
            if (element) {
                element.textContent = "--";
            }
        });
    }


    // ==========================================
    // 8. WEBSITE PREVIEW RESET
    // ==========================================

    function resetWebsitePreview() {
        if (previewStatus) {
            previewStatus.textContent = "AWAITING PREVIEW";
        }

        if (previewDomain) {
            previewDomain.textContent = "Website preview";
        }

        if (websiteScreenshot) {
            websiteScreenshot.onload = null;
            websiteScreenshot.onerror = null;

            websiteScreenshot.removeAttribute("src");
            websiteScreenshot.style.display = "none";
            websiteScreenshot.hidden = true;

            websiteScreenshot.alt =
                "Screenshot of the inspected website";
        }

        if (websitePreviewPlaceholder) {
            websitePreviewPlaceholder.replaceChildren();

            const icon = document.createElement("div");
            icon.className = "preview-placeholder-icon";
            icon.textContent = "◎";

            const heading = document.createElement("strong");
            heading.textContent =
                "Website preview will appear here";

            const description = document.createElement("p");
            description.textContent =
                "A screenshot will be displayed when the website " +
                "can be captured successfully.";

            websitePreviewPlaceholder.append(
                icon,
                heading,
                description
            );

            websitePreviewPlaceholder.hidden = false;
            websitePreviewPlaceholder.style.display = "flex";
        }

        if (previewPageTitle) {
            previewPageTitle.textContent = "Not available";
        }

        if (previewReachability) {
            previewReachability.textContent = "Not checked";
        }
    }


    // ==========================================
    // 9. WEBSITE PREVIEW ERROR DISPLAY
    // ==========================================

    function showPreviewFailure(message) {
        if (websiteScreenshot) {
            websiteScreenshot.onload = null;
            websiteScreenshot.onerror = null;

            websiteScreenshot.removeAttribute("src");
            websiteScreenshot.style.display = "none";
            websiteScreenshot.hidden = true;
        }

        if (websitePreviewPlaceholder) {
            websitePreviewPlaceholder.replaceChildren();

            const heading = document.createElement("strong");
            heading.textContent = "Preview unavailable";

            const description = document.createElement("p");
            description.textContent = message;

            websitePreviewPlaceholder.append(
                heading,
                description
            );

            websitePreviewPlaceholder.hidden = false;
            websitePreviewPlaceholder.style.display = "flex";
        }

        if (previewStatus) {
            previewStatus.textContent = "PREVIEW UNAVAILABLE";
        }

        if (previewPageTitle) {
            previewPageTitle.textContent = "Not available";
        }

        if (previewReachability) {
            previewReachability.textContent = "Could not verify";
        }
    }


    // ==========================================
    // 10. CAPTURE AND DISPLAY WEBSITE SCREENSHOT
    // ==========================================

    async function fetchWebsitePreview(url) {
        if (previewStatus) {
            previewStatus.textContent =
                "CAPTURING WEBSITE SCREENSHOT...";
        }

        if (previewDomain) {
            try {
                previewDomain.textContent =
                    new URL(url).hostname;
            } catch {
                previewDomain.textContent = "Website preview";
            }
        }

        if (websiteScreenshot) {
            websiteScreenshot.onload = null;
            websiteScreenshot.onerror = null;

            websiteScreenshot.removeAttribute("src");
            websiteScreenshot.style.display = "none";
            websiteScreenshot.hidden = true;
        }

        if (websitePreviewPlaceholder) {
            websitePreviewPlaceholder.replaceChildren();

            const message = document.createElement("p");
            message.textContent =
                "Loading website screenshot...";

            websitePreviewPlaceholder.append(message);

            websitePreviewPlaceholder.hidden = false;
            websitePreviewPlaceholder.style.display = "flex";
        }

        try {
            const response = await fetch("/website-preview", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                    "Accept": "application/json"
                },
                body: JSON.stringify({ url })
            });

            let data;

            try {
                data = await response.json();
            } catch {
                throw new Error(
                    `Invalid preview response (HTTP ${response.status}).`
                );
            }

            if (!response.ok) {
                throw new Error(
                    data.error ||
                    `Preview request failed (HTTP ${response.status}).`
                );
            }

            /*
             * Accepted screenshot formats:
             * 1. data:image/jpeg;base64,...
             * 2. data:image/png;base64,...
             * 3. Raw Base64 screenshot data
             */

            if (
                typeof data.screenshot !== "string" ||
                !data.screenshot.trim()
            ) {
                throw new Error(
                    data.error ||
                    "The server did not return a screenshot."
                );
            }

            let screenshotSrc = data.screenshot.trim();

            if (!screenshotSrc.startsWith("data:image/")) {
                screenshotSrc =
                    "data:image/jpeg;base64," + screenshotSrc;
            }

            if (!websiteScreenshot) {
                throw new Error(
                    "The HTML image element #websiteScreenshot was not found."
                );
            }

            // Load the image before displaying it.
            await new Promise((resolve, reject) => {
                websiteScreenshot.onload = () => resolve();

                websiteScreenshot.onerror = () => {
                    reject(
                        new Error(
                            "The screenshot was received but could not be displayed."
                        )
                    );
                };

                websiteScreenshot.src = screenshotSrc;
            });

            websiteScreenshot.hidden = false;
            websiteScreenshot.style.display = "block";

            if (websitePreviewPlaceholder) {
                websitePreviewPlaceholder.hidden = true;
                websitePreviewPlaceholder.style.display = "none";
            }

            if (previewStatus) {
                previewStatus.textContent = "SCREENSHOT CAPTURED";
            }

            if (previewDomain) {
                previewDomain.textContent =
                    data.domain || new URL(url).hostname;
            }

            if (previewPageTitle) {
                previewPageTitle.textContent =
                    data.page_title || "Title unavailable";
            }

            if (previewReachability) {
                const statusCode = data.http_status
                    ? ` (HTTP ${data.http_status})`
                    : "";

                previewReachability.textContent =
                    data.reachable === true
                        ? `Responded${statusCode}`
                        : `Response could not be confirmed${statusCode}`;
            }

            console.log(
                "Website screenshot captured successfully:",
                data.domain || url
            );

        } catch (error) {
            console.error(
                "Website preview failed:",
                error
            );

            showPreviewFailure(
                error.message ||
                "The website screenshot could not be captured."
            );
        }
    }


    // ==========================================
    // 11. RESULT MODAL HELPERS
    // ==========================================

    function openResultModal() {
        if (!resultModal) {
            return;
        }

        resultModal.style.display = "flex";

        resultModal.setAttribute("aria-hidden", "false");
        resultModal.setAttribute("aria-modal", "true");
        resultModal.setAttribute("role", "dialog");

        document.body.style.overflow = "hidden";

        if (closeModal) {
            closeModal.focus();
        }
    }

    function closeResult() {
        if (resultModal) {
            resultModal.style.display = "none";
            resultModal.setAttribute("aria-hidden", "true");
            resultModal.removeAttribute("aria-modal");
        }

        document.body.style.overflow = "";

        if (scanButton && !scanButton.disabled) {
            scanButton.focus();
        } else if (urlInput && !urlInput.disabled) {
            urlInput.focus();
        }
    }


    // ==========================================
    // 12. DISPLAY PREDICTION RESULTS
    // ==========================================

    function displayResult(data, url) {
        if (!data || typeof data !== "object") {
            throw new Error(
                "The server did not return a valid prediction."
            );
        }

        /*
         * Existing project convention:
         * 0 = Legitimate
         * 1 = Phishing
         */

        const prediction = Number(data.prediction);

        const label = String(
            data.label ||
            (
                prediction === 0
                    ? "Legitimate"
                    : prediction === 1
                        ? "Phishing"
                        : "Unknown"
            )
        ).trim();

        const normalizedLabel = label.toLowerCase();

        const isLegitimate =
            normalizedLabel === "legitimate";

        const isPhishing =
            normalizedLabel === "phishing";

        const isKnownLabel =
            isLegitimate || isPhishing;

        if (modalIcon) {
            modalIcon.textContent = isLegitimate
                ? "✓"
                : isPhishing
                    ? "!"
                    : "?";
        }

        if (modalStatus) {
            modalStatus.textContent = isLegitimate
                ? "MODEL CLASSIFICATION: LEGITIMATE"
                : isPhishing
                    ? "MODEL CLASSIFICATION: PHISHING"
                    : "CLASSIFICATION UNAVAILABLE";
        }

        if (modalTitle) {
            modalTitle.textContent = isLegitimate
                ? "Classified as Legitimate"
                : isPhishing
                    ? "Potential Phishing Website"
                    : "Unable to Determine Result";
        }

        if (modalDescription) {
            modalDescription.textContent = isLegitimate
                ? "The primary model classified this URL as legitimate. " +
                  "This prediction does not guarantee that the website is safe."
                : isPhishing
                    ? "The primary model classified this URL as potentially " +
                      "associated with phishing. Avoid entering sensitive " +
                      "information until the website is independently verified."
                    : "The system did not return a recognized classification. " +
                      "Treat the result as inconclusive.";
        }

        if (checkedUrl) {
            checkedUrl.textContent = url;
            checkedUrl.title = url;
        }

        resetModelResults();

        const models = data.models || {};

        displayModelResult(
            randomForestCard,
            randomForestResult,
            randomForestConfidence,
            models.random_forest
        );

        displayModelResult(
            logisticCard,
            logisticResult,
            logisticConfidence,
            models.logistic_regression
        );

        displayModelResult(
            naiveBayesCard,
            naiveBayesResult,
            naiveBayesConfidence,
            models.naive_bayes
        );

        // Check agreement among the three model predictions.
        const modelEntries = [
            models.random_forest,
            models.logistic_regression,
            models.naive_bayes
        ];

        const validLabels = modelEntries
            .filter(model =>
                model &&
                ["legitimate", "phishing"].includes(
                    String(model.label || "")
                        .trim()
                        .toLowerCase()
                )
            )
            .map(model =>
                String(model.label)
                    .trim()
                    .toLowerCase()
            );

        const distinctLabels = new Set(validLabels);

        if (
            modalDescription &&
            distinctLabels.size > 1
        ) {
            modalDescription.textContent =
                "The machine learning models disagree on this URL. " +
                "Review each prediction carefully and treat the overall " +
                "classification with caution.";

        } else if (
            modalDescription &&
            validLabels.length === 3 &&
            distinctLabels.size === 1
        ) {
            modalDescription.textContent =
                `All three models classified this URL as ${validLabels[0]}. ` +
                "Agreement does not guarantee that the website is safe.";

        } else if (
            modalDescription &&
            !isKnownLabel
        ) {
            modalDescription.textContent =
                "The system could not determine a reliable overall " +
                "classification. Review the available model results.";
        }

        openResultModal();
    }


    // ==========================================
    // 13. SCAN URL
    // ==========================================

    async function scanUrl() {
        if (
            !urlInput ||
            !scanButton ||
            scanButton.disabled
        ) {
            return;
        }

        hideError();

        const input = urlInput.value.trim();

        if (!input) {
            showError("Please enter a website URL.");
            urlInput.focus();
            return;
        }

        const url = normalizeUrl(input);

        if (!isValidUrl(url)) {
            showError(
                "Enter a valid HTTP or HTTPS website address, " +
                "such as example.com or https://example.com/page."
            );

            urlInput.focus();
            return;
        }

        showLoading();

        resetModelResults();
        resetWebsitePreview();

        try {
            // Start both requests together.
            const predictionPromise = fetch("/predict", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                    "Accept": "application/json"
                },
                body: JSON.stringify({ url })
            });

            // Screenshot errors are handled independently.
            const previewPromise =
                fetchWebsitePreview(url);

            const response = await predictionPromise;

            let data;

            try {
                data = await response.json();
            } catch {
                throw new Error(
                    `The prediction server returned an invalid response ` +
                    `(HTTP ${response.status}).`
                );
            }

            if (!response.ok || data.error) {
                throw new Error(
                    data.error ||
                    `Prediction failed (HTTP ${response.status}).`
                );
            }

            // Show the ML predictions even if preview capture fails.
            displayResult(data, url);

            // Wait for screenshot capture before enabling another scan.
            await previewPromise;

        } catch (error) {
            console.error(
                "PhishGuard scan error:",
                error
            );

            showError(
                error && error.message
                    ? error.message
                    : "Unable to analyze this URL. Please try again."
            );

        } finally {
            hideLoading();
        }
    }


    // ==========================================
    // 14. EVENT LISTENERS
    // ==========================================

    if (scanButton) {
        scanButton.addEventListener(
            "click",
            scanUrl
        );
    }

    if (urlInput) {
        urlInput.addEventListener("keydown", event => {
            if (event.key === "Enter") {
                event.preventDefault();
                scanUrl();
            }
        });

        urlInput.addEventListener(
            "input",
            hideError
        );
    }

    if (closeModal) {
        closeModal.addEventListener(
            "click",
            closeResult
        );
    }

    if (modalBackdrop) {
        modalBackdrop.addEventListener(
            "click",
            closeResult
        );
    }

    if (newScanButton) {
        newScanButton.addEventListener("click", () => {
            closeResult();

            resetModelResults();
            resetWebsitePreview();
            hideError();

            if (urlInput) {
                urlInput.value = "";
                urlInput.focus();
            }
        });
    }

    document.addEventListener("keydown", event => {
        if (
            event.key === "Escape" &&
            resultModal &&
            resultModal.style.display !== "none"
        ) {
            closeResult();
        }
    });

});
