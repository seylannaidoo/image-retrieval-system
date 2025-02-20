document.addEventListener('DOMContentLoaded', function() {
    // Form elements
    const form = document.getElementById('searchForm');
    const topKSlider = document.getElementById('top_k');
    const topKValue = document.getElementById('top_k_value');
    const resultsDiv = document.getElementById('results');
    const loadingDiv = document.getElementById('loading');
    const errorDiv = document.getElementById('error');
    const searchInput = document.getElementById('query');

    // Night mode control
    const nightModeToggle = document.getElementById('nightModeToggle');

    // Load user preferences from localStorage
    const loadPreferences = () => {
        // Only enable night mode if explicitly set by user
        const nightMode = localStorage.getItem('nightMode') === 'true';
        if (nightMode) {
            document.body.classList.add('night-mode');
            nightModeToggle.setAttribute('aria-pressed', 'true');
        }
    };

    // Optional: Initialize accessibility features
    loadPreferences();

    // Night mode toggle
    nightModeToggle?.addEventListener('click', () => {
        document.body.classList.toggle('night-mode');
        const isPressed = document.body.classList.contains('night-mode');
        nightModeToggle.setAttribute('aria-pressed', isPressed);
        localStorage.setItem('nightMode', isPressed);
    });

    // Update slider value display
    topKSlider.addEventListener('input', function() {
        topKValue.textContent = this.value;
        topKSlider.setAttribute('aria-valuenow', this.value);
        // Announce change to screen readers
        const announcement = `Number of results set to ${this.value}`;
        announceToScreenReader(announcement);
    });

    // Helper function for screen reader announcements
    function announceToScreenReader(message) {
        const announcement = document.createElement('div');
        announcement.setAttribute('role', 'status');
        announcement.setAttribute('aria-live', 'polite');
        announcement.className = 'sr-only';
        announcement.textContent = message;
        document.body.appendChild(announcement);
        setTimeout(() => announcement.remove(), 1000);
    }

    // Optional: Keyboard navigation for results
    const handleResultsKeyboard = (event) => {
        const cards = resultsDiv.querySelectorAll('.result-card');
        const currentCard = document.activeElement;

        if (!currentCard.classList.contains('result-card')) return;

        const index = Array.from(cards).indexOf(currentCard);
        let nextIndex;

        switch(event.key) {
            case 'ArrowRight':
                nextIndex = Math.min(index + 1, cards.length - 1);
                break;
            case 'ArrowLeft':
                nextIndex = Math.max(index - 1, 0);
                break;
            case 'ArrowDown':
                nextIndex = Math.min(index + 3, cards.length - 1);
                break;
            case 'ArrowUp':
                nextIndex = Math.max(index - 3, 0);
                break;
            case 'Home':
                nextIndex = 0;
                break;
            case 'End':
                nextIndex = cards.length - 1;
                break;
            default:
                return;
        }

        cards[nextIndex].focus();
        event.preventDefault();
    };

    resultsDiv.addEventListener('keydown', handleResultsKeyboard);

    // Handle form submission
    form.addEventListener('submit', async function(e) {
        e.preventDefault();

        // Clear previous results and errors
        resultsDiv.innerHTML = '';
        errorDiv.style.display = 'none';
        loadingDiv.style.display = 'block';

        // Announce loading state to screen readers
        announceToScreenReader('Searching for images...');

        const formData = new FormData(form);

        try {
            const response = await fetch('/search', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.error || 'Search failed');
            }

            // Display results
            data.results.forEach((result, index) => {
                const card = document.createElement('div');
                card.className = 'result-card';
                card.tabIndex = 0; // Make focusable for keyboard navigation
                card.setAttribute('role', 'article');
                card.setAttribute('aria-label', `Image ${index + 1} of ${data.results.length}: ${result.filename} with ${result.similarity} similarity`);

                card.innerHTML = `
                    <img src="data:image/jpeg;base64,${result.image}" 
                         alt="Search result showing ${result.filename}"
                         loading="lazy">
                    <div class="result-info">
                        <div class="similarity" aria-label="Similarity score: ${result.similarity}">
                            ${result.similarity}
                        </div>
                        <div class="filename">${result.filename}</div>
                    </div>
                `;

                // Add keyboard event for Enter key
                card.addEventListener('keydown', (e) => {
                    if (e.key === 'Enter') {
                        // You could add functionality here to show a larger version
                        // of the image or more details
                        announceToScreenReader(`Selected image: ${result.filename}`);
                    }
                });

                resultsDiv.appendChild(card);
            });

            // Announce results to screen readers
            const resultCount = data.results.length;
            const announcement = `Found ${resultCount} matching ${resultCount === 1 ? 'image' : 'images'}`;
            announceToScreenReader(announcement);

            // Focus the first result if available
            if (resultCount > 0) {
                const firstCard = resultsDiv.querySelector('.result-card');
                firstCard?.focus();
            }

        } catch (error) {
            errorDiv.textContent = error.message;
            errorDiv.style.display = 'block';
            announceToScreenReader(`Error: ${error.message}`);
        } finally {
            loadingDiv.style.display = 'none';
        }
    });

    // Optional: Add escape key handler to return focus to search
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') {
            searchInput.focus();
        }
    });
});