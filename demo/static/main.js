document.addEventListener('DOMContentLoaded', () => {
    // UI Elements
    const tabButtons = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');
    const configForm = document.getElementById('config-form');
    
    // Sliders & Range Values
    const chunkSizeSlider = document.getElementById('chunk_size');
    const chunkOverlapSlider = document.getElementById('chunk_overlap');
    const temperatureSlider = document.getElementById('temperature');
    const sizeVal = document.getElementById('size-val');
    const overlapVal = document.getElementById('overlap-val');
    const tempVal = document.getElementById('temp-val');
    
    // Select & Inputs
    const strategySelect = document.getElementById('strategy');
    const strategyHelp = document.getElementById('strategy-help');
    const modelInput = document.getElementById('model');
    const enrichToggle = document.getElementById('enrich');
    const enrichmentDetails = document.getElementById('enrichment-details');
    const presetTags = document.querySelectorAll('.preset-tag');
    
    // Text Tab
    const rawTextArea = document.getElementById('raw-text');
    const loadSampleBtn = document.getElementById('load-sample-btn');
    
    // File Tab / Dropzone
    const dropzone = document.getElementById('dropzone');
    const fileInput = document.getElementById('file-input');
    const dropzonePrompt = dropzone.querySelector('.dropzone-prompt');
    const fileInfo = dropzone.querySelector('.dropzone-file-info');
    const fileNameLabel = document.getElementById('file-name-label');
    const removeFileBtn = document.getElementById('remove-file');
    
    // Overlay & Results
    const loadingOverlay = document.getElementById('loading-overlay');
    const loadingStep = document.getElementById('loading-step');
    const progressFill = document.getElementById('progress-fill');
    const consoleLogs = document.getElementById('status-console');
    const resultsPanel = document.getElementById('results-panel');
    const chunksList = document.getElementById('chunks-list');
    
    // Metrics
    const statChunks = document.getElementById('stat-chunks');
    const statTime = document.getElementById('stat-time');
    const statTokens = document.getElementById('stat-tokens');
    const statCost = document.getElementById('stat-cost');
    
    // Cache Stats
    const cacheHitsEl = document.getElementById('cache-hits');
    const cacheMissesEl = document.getElementById('cache-misses');
    const cacheHitRateEl = document.getElementById('cache-hit-rate');
    const cacheTimeSavedEl = document.getElementById('cache-time-saved');
    const cacheEntriesEl = document.getElementById('cache-entries');
    const cacheCallsSavedEl = document.getElementById('cache-calls-saved');
    
    // Export Actions
    const copyJsonBtn = document.getElementById('copy-json-btn');
    const downloadJsonBtn = document.getElementById('download-json-btn');
    const viewRawBtn = document.getElementById('view-raw-btn');
    
    // Modal
    const rawModal = document.getElementById('raw-modal');
    const rawJsonOutput = document.getElementById('raw-json-output');
    const closeModalBtn = document.getElementById('close-modal');
    const modalCloseBtn = document.getElementById('modal-close-btn');
    const modalCopyBtn = document.getElementById('modal-copy-btn');
    
    // State variables
    let currentTab = 'text-tab';
    let selectedFile = null;
    let processedChunksData = null; // Stored parsed chunk response
    
    // Strategy descriptions helper
    const STRATEGY_HELPER = {
        recursive: 'Splits on paragraph, line, and sentence delimiters recursively to fit token limits.',
        semantic: 'Embeds sentences and splits where adjacent sentences diverge in topic similarity.',
        structural: 'Uses Markdown headings (#) as chunk boundaries, falls back to recursive chunking for large sections.'
    };

    // Sample Text
    const SAMPLE_DOC = `# SmartChunk Architecture Design

## Executive Summary
SmartChunk is a Python library designed to optimize Retrieval-Augmented Generation (RAG) by embedding document chunks with critical contextual metadata. By enriching every text chunk with summaries, entities, keywords, parent hierarchies, and neighbor linkages, SmartChunk provides a 10x retrieval signal boost over raw character-based splitters.

## Technical Details
Most vector databases suffer from "context fragmentation". If an LLM response spans multiple chunks, a retriever might miss key context. SmartChunk solves this by wrapping text splitting inside a multi-stage parser and enricher pipeline:

1. **Parser**: ingests raw PDF, Markdown, or text, extracting structured sections.
2. **Chunker**: splits text using recursive, structural, or embedding-based semantic boundaries.
3. **Enricher**: queries a LiteLLM model to attach semantic metadata to every chunk in parallel.
4. **Exporter**: writes out to Pinecone, ChromaDB, JSON, or JSONL.

## Roadmap & Budget
The SmartChunk contributors have allocated a $150K budget for development in Q3 2026. Key targets include local model acceleration, tree-based hierarchical indexing, and native integrations with LlamaIndex. The project is led by lead architect Sarah Jenkins.`;

    // Initialize Ollama status check
    checkOllamaStatus();

    // ── Tab Management ────────────────────────────────────────────────────────
    tabButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            tabButtons.forEach(b => b.classList.remove('active'));
            tabContents.forEach(c => c.classList.remove('active'));
            
            btn.classList.add('active');
            const tabId = btn.getAttribute('data-tab');
            document.getElementById(tabId).classList.add('active');
            currentTab = tabId;
        });
    });

    // ── Sliders & Controls ───────────────────────────────────────────────────
    chunkSizeSlider.addEventListener('input', (e) => {
        sizeVal.textContent = e.target.value;
    });
    
    chunkOverlapSlider.addEventListener('input', (e) => {
        overlapVal.textContent = e.target.value;
    });
    
    temperatureSlider.addEventListener('input', (e) => {
        tempVal.textContent = parseFloat(e.target.value).toFixed(1);
    });

    strategySelect.addEventListener('change', (e) => {
        const val = e.target.value;
        strategyHelp.textContent = STRATEGY_HELPER[val] || '';
    });

    // Toggle LLM details panel
    enrichToggle.addEventListener('change', (e) => {
        if (e.target.checked) {
            enrichmentDetails.classList.add('expanded');
        } else {
            enrichmentDetails.classList.remove('expanded');
        }
    });

    // Presets
    presetTags.forEach(tag => {
        tag.addEventListener('click', () => {
            presetTags.forEach(t => t.classList.remove('active'));
            tag.classList.add('active');
            modelInput.value = tag.getAttribute('data-model');
        });
    });

    // Custom model input resets presets
    modelInput.addEventListener('input', () => {
        presetTags.forEach(t => t.classList.remove('active'));
    });

    // Load sample text
    loadSampleBtn.addEventListener('click', () => {
        rawTextArea.value = SAMPLE_DOC;
        logConsole("Loaded architecture sample document.");
    });

    // ── File Upload / Dropzone ───────────────────────────────────────────────
    
    // Highlight drop area on drag
    ['dragenter', 'dragover'].forEach(eventName => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            dropzone.classList.add('dragover');
        }, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            dropzone.classList.remove('dragover');
        }, false);
    });

    // Handle dropped files
    dropzone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files.length > 0) {
            handleSelectedFile(files[0]);
        }
    });

    // File input selection
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleSelectedFile(e.target.files[0]);
        }
    });

    // Browse files click delegation
    dropzone.querySelector('.select-file-btn').addEventListener('click', (e) => {
        e.stopPropagation();
        fileInput.click();
    });

    removeFileBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        clearFileSelection();
    });

    function handleSelectedFile(file) {
        selectedFile = file;
        fileNameLabel.textContent = file.name;
        dropzonePrompt.style.display = 'none';
        fileInfo.style.display = 'flex';
        logConsole(`Selected file: ${file.name} (${formatBytes(file.size)})`);
    }

    function clearFileSelection() {
        selectedFile = null;
        fileInput.value = '';
        dropzonePrompt.style.display = 'block';
        fileInfo.style.display = 'none';
        logConsole("Cleared file selection.");
    }

    // ── Ollama Connection Probe ──────────────────────────────────────────────
    async function checkOllamaStatus() {
        const indicator = document.getElementById('ollama-indicator');
        const label = document.getElementById('ollama-label');
        
        try {
            const res = await fetch('http://localhost:11434/api/tags');
            if (res.ok) {
                const data = await res.json();
                indicator.className = "indicator green";
                const models = data.models || [];
                const names = models.map(m => m.name).slice(0, 2);
                const suffix = models.length > 2 ? ` (+${models.length - 2} more)` : '';
                label.textContent = models.length > 0 
                    ? `Ollama: Connected [${names.join(', ')}${suffix}]` 
                    : "Ollama: Connected (No models found)";
                logConsole(`Discovered local Ollama models: ${models.map(m => m.name).join(', ')}`);
            } else {
                throw new Error("Bad response");
            }
        } catch (e) {
            indicator.className = "indicator gray";
            label.textContent = "Ollama: Offline";
            logConsole("Local Ollama endpoint (http://localhost:11434) is currently unreachable. Select OpenAI if you don't have local LLMs running.");
        }
    }

    // ── Logging / Console ─────────────────────────────────────────────────────
    function logConsole(message, type = 'info') {
        const line = document.createElement('div');
        line.className = `console-line ${type}`;
        
        const timestamp = new Date().toLocaleTimeString();
        line.textContent = `[${timestamp}] [${type.toUpperCase()}] ${message}`;
        
        consoleLogs.appendChild(line);
        consoleLogs.scrollTop = consoleLogs.scrollHeight;
    }

    function clearConsole() {
        consoleLogs.innerHTML = '';
    }

    // ── Form Submit & API Pipeline Call ──────────────────────────────────────
    configForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        // Validation — only block if active tab is text or file
        if (currentTab === 'text-tab' && !rawTextArea.value.trim()) {
            alert("Please paste some text first.");
            return;
        }
        if (currentTab === 'file-tab' && !selectedFile) {
            alert("Please select or drop a file first.");
            return;
        }
        // If cost tab is active, default to text tab behavior
        if (currentTab === 'cost-tab') {
            if (!rawTextArea.value.trim() && !selectedFile) {
                alert("Please paste text or upload a file first.");
                return;
            }
        }

        // Setup loading state
        clearConsole();
        resultsPanel.style.display = 'none';
        loadingOverlay.style.display = 'flex';
        loadingStep.textContent = "Parsing configuration...";
        progressFill.style.width = '10%';

        // Collect fields
        const strategy = strategySelect.value;
        const chunkSize = parseInt(chunkSizeSlider.value);
        const chunkOverlap = parseInt(chunkOverlapSlider.value);
        const enrich = enrichToggle.checked;
        const model = modelInput.value;
        const temperature = parseFloat(temperatureSlider.value);
        
        // Enrich fields checkbox extraction
        const enrichFields = [];
        document.querySelectorAll('input[name="enrich_fields"]:checked').forEach(cb => {
            enrichFields.push(cb.value);
        });

        logConsole("Initializing pipeline settings...");
        logConsole(`Strategy: ${strategy} | Chunk Size: ${chunkSize} tokens | Overlap: ${chunkOverlap} tokens`);
        logConsole(`LLM Enrichment: ${enrich ? 'ON' : 'OFF'} (${enrich ? 'Model: ' + model : ''})`);
        logConsole(`Cost Optimization: Hash-based caching ENABLED`);

        let responsePromise;
        
        // Determine data source
        const useFile = (currentTab === 'file-tab') || (currentTab === 'cost-tab' && selectedFile);

        if (!useFile) {
            loadingStep.textContent = "Sending raw text to parser...";
            progressFill.style.width = '40%';
            logConsole("Parsing and normalising raw text strings...");
            
            responsePromise = fetch('/api/process/text', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    text: rawTextArea.value,
                    source: "web_dashboard_input.txt",
                    strategy: strategy,
                    chunk_size: chunkSize,
                    chunk_overlap: chunkOverlap,
                    enrich: enrich,
                    model: model,
                    temperature: temperature,
                    enrichments: enrichFields
                })
            });
        } else {
            loadingStep.textContent = "Uploading document file...";
            progressFill.style.width = '30%';
            logConsole(`Uploading ${selectedFile.name} to FastAPI backend...`);
            
            const formData = new FormData();
            formData.append('file', selectedFile);
            formData.append('strategy', strategy);
            formData.append('chunk_size', chunkSize);
            formData.append('chunk_overlap', chunkOverlap);
            formData.append('enrich', enrich);
            formData.append('model', model);
            formData.append('temperature', temperature);
            formData.append('enrichments', enrichFields.join(','));
            
            responsePromise = fetch('/api/process/file', {
                method: 'POST',
                body: formData
            });
        }

        try {
            logConsole("Executing text chunker partitions...");
            loadingStep.textContent = "Splitting document into token-aware partitions...";
            progressFill.style.width = '60%';
            
            if (enrich) {
                logConsole("Contacting LLM for chunk summaries, entities, and keywords...");
                logConsole("Hash-based cache will skip duplicate chunk texts...");
                loadingStep.textContent = "Running async parallel LLM enrichment with caching...";
                progressFill.style.width = '80%';
            }

            const response = await responsePromise;
            const data = await response.json();
            
            if (!data.success) {
                throw new Error(data.detail || "Server error occurred");
            }
            
            loadingStep.textContent = "Linking sequential neighbor summaries...";
            progressFill.style.width = '95%';
            logConsole("Assembling self-contained SmartChunk payloads...");

            // Log cache performance
            if (data.stats.cache) {
                const c = data.stats.cache;
                logConsole(`Cache Report: ${c.cache_hits} hits, ${c.cache_misses} misses, ${c.hit_rate_percent}% hit rate`);
                if (c.llm_calls_saved > 0) {
                    logConsole(`💰 Saved ${c.llm_calls_saved} LLM calls (~${c.estimated_time_saved_seconds}s estimated time saved)`);
                }
            }
            
            processedChunksData = data; // Store globally
            
            // Short timeout to let the user see the progress bar hit 100%
            setTimeout(() => {
                loadingOverlay.style.display = 'none';
                displayResults(data);
            }, 600);

        } catch (err) {
            logConsole(`Error: ${err.message}`, 'error');
            loadingStep.textContent = "Failed processing pipeline";
            progressFill.style.width = '100%';
            progressFill.style.backgroundColor = 'var(--accent-rose)';
            
            // Add a close button or clear state
            setTimeout(() => {
                if (confirm("Processing failed. Close loading state?")) {
                    loadingOverlay.style.display = 'none';
                }
            }, 1000);
        }
    });

    // ── Render Results Dashboard ──────────────────────────────────────────────
    function displayResults(data) {
        // Toggle view
        resultsPanel.style.display = 'flex';
        
        // Update stats
        statChunks.textContent = data.stats.total_chunks;
        statTime.textContent = `${data.stats.duration_seconds}s`;
        
        const usage = data.stats.usage;
        statTokens.textContent = usage.total_tokens > 0 ? usage.total_tokens : 'N/A';
        statCost.textContent = usage.estimated_cost_usd > 0 ? `$${usage.estimated_cost_usd.toFixed(4)}` : '$0.00';
        
        // Update cache stats
        const cache = data.stats.cache || {};
        if (cacheHitsEl) cacheHitsEl.textContent = cache.cache_hits || 0;
        if (cacheMissesEl) cacheMissesEl.textContent = cache.cache_misses || 0;
        if (cacheHitRateEl) cacheHitRateEl.textContent = `${cache.hit_rate_percent || 0}%`;
        if (cacheTimeSavedEl) cacheTimeSavedEl.textContent = `${cache.estimated_time_saved_seconds || 0}s`;
        if (cacheEntriesEl) cacheEntriesEl.textContent = cache.cache_entries || 0;
        if (cacheCallsSavedEl) cacheCallsSavedEl.textContent = cache.llm_calls_saved || 0;

        // Animate cache stat boxes
        document.querySelectorAll('.cache-stat-box').forEach((box, i) => {
            box.style.opacity = '0';
            box.style.transform = 'translateY(10px)';
            setTimeout(() => {
                box.style.transition = 'opacity 0.4s ease, transform 0.4s ease';
                box.style.opacity = '1';
                box.style.transform = 'translateY(0)';
            }, 100 + i * 80);
        });

        // Render chunks
        chunksList.innerHTML = '';
        
        data.chunks.forEach((chunk, idx) => {
            const card = document.createElement('div');
            card.className = 'chunk-card card glass';
            
            // Build Parent Context badge
            const parentContextHtml = chunk.parent_context 
                ? `<div class="chunk-parent-path">📂 ${chunk.parent_context}</div>`
                : '';
                
            // Check entities & keywords
            let entitiesHtml = chunk.entities.map(e => `<span class="badge entity">${escapeHtml(e)}</span>`).join('');
            if (!entitiesHtml) entitiesHtml = '<span class="no-tags">No entities extracted</span>';
            
            let keywordsHtml = chunk.keywords.map(k => `<span class="badge keyword">${escapeHtml(k)}</span>`).join('');
            if (!keywordsHtml) keywordsHtml = '<span class="no-tags">No keywords extracted</span>';

            // Confidence progress bar
            const confidenceVal = parseFloat(chunk.confidence);
            let confidenceClass = 'low';
            if (confidenceVal >= 0.8) confidenceClass = 'high';
            else if (confidenceVal >= 0.5) confidenceClass = 'medium';
            
            const confidenceHtml = chunk.confidence > 0 
                ? `
                <div class="enrich-section">
                    <span class="enrich-label">Atomicity / Confidence</span>
                    <div class="confidence-indicator">
                        <div class="confidence-bar-container">
                            <div class="confidence-bar-fill ${confidenceClass}" style="width: ${confidenceVal * 100}%"></div>
                        </div>
                        <span class="confidence-value ${confidenceClass}">${confidenceVal.toFixed(2)}</span>
                    </div>
                </div>`
                : '';
                
            // Neighbor Linking
            const prevText = chunk.prev_summary ? escapeHtml(chunk.prev_summary) : '(Start of document)';
            const nextText = chunk.next_summary ? escapeHtml(chunk.next_summary) : '(End of document)';
            const neighborChainHtml = (chunk.prev_summary || chunk.next_summary)
                ? `
                <div class="enrich-section">
                    <span class="enrich-label">Context Neighbors</span>
                    <div class="neighbor-chain">
                        <div class="neighbor-node" title="${prevText}">
                            <span class="neighbor-title">⬅️ Prev Summary</span>
                            <span class="neighbor-text">${prevText}</span>
                        </div>
                        <div class="neighbor-node" title="${nextText}">
                            <span class="neighbor-title">Next Summary ➡️</span>
                            <span class="neighbor-text">${nextText}</span>
                        </div>
                    </div>
                </div>`
                : '';

            // Summary
            const summaryHtml = chunk.summary 
                ? `
                <div class="enrich-section">
                    <span class="enrich-label">LLM Summary</span>
                    <p class="chunk-summary-box">"${escapeHtml(chunk.summary)}"</p>
                </div>`
                : '';

            // Contextual Embedding Payload Box
            let cacheBadgeClass = chunk.cache_status === 'HIT' ? 'hit' : 'miss';
            if (chunk.cache_status === 'DISABLED') cacheBadgeClass = 'disabled';

            const contextualHtml = chunk.contextual_text 
                ? `
                <div class="contextual-preview-box">
                    <span class="contextual-label">🎯 Contextual Embedding Payload</span>
                    <code class="contextual-code">${escapeHtml(chunk.contextual_text)}</code>
                    <div class="embedding-meta-row">
                        <div class="embedding-meta-item">
                            <span class="embedding-meta-label">Embedding Model</span>
                            <span class="embedding-meta-value">${escapeHtml(chunk.embedding_model || 'all-MiniLM-L6-v2')}</span>
                        </div>
                        <div class="embedding-meta-item">
                            <span class="embedding-meta-label">Dimensions</span>
                            <span class="embedding-meta-value">${chunk.embedding_dimensions || 384}</span>
                        </div>
                        <div class="embedding-meta-item">
                            <span class="embedding-meta-label">Cache Status</span>
                            <span class="embedding-meta-value badge-cache-${cacheBadgeClass}">${escapeHtml(chunk.cache_status)}</span>
                        </div>
                        <div class="embedding-meta-item">
                            <span class="embedding-meta-label">Chunk Hash</span>
                            <span class="embedding-meta-value text-monospace">${escapeHtml(chunk.id.replace('chunk_', '').substring(0, 8))}</span>
                        </div>
                    </div>
                </div>`
                : '';

            // Figures Rendering
            let figuresHtml = '';
            if (chunk.figures && chunk.figures.length > 0) {
                figuresHtml = `
                <div class="enrich-section">
                    <span class="enrich-label">🖼️ Associated Figures</span>
                    <div class="figures-list">
                        ${chunk.figures.map(fig => `
                            <div class="figure-item">
                                <span class="figure-icon">📊</span>
                                <div class="figure-details">
                                    <span class="figure-caption">${escapeHtml(fig.caption || 'Unnamed Figure')}</span>
                                    <span class="figure-meta">Page ${fig.page || 'N/A'} ${fig.bbox ? `| BBox: [${fig.bbox.map(n => Math.round(n)).join(', ')}]` : ''}</span>
                                </div>
                            </div>
                        `).join('')}
                    </div>
                </div>`;
            }

            // Table Rendering
            let bodyTextHtml = `<code class="raw-code">${escapeHtml(chunk.text)}</code>`;
            if (chunk.content_type === 'table' && chunk.table && chunk.table.rows && chunk.table.rows.length > 0) {
                const headers = chunk.table.headers || [];
                const rows = chunk.table.rows || [];
                let ths = headers.map(h => `<th>${escapeHtml(h)}</th>`).join('');
                let trs = rows.map(r => `<tr>${r.map(c => `<td>${escapeHtml(c)}</td>`).join('')}</tr>`).join('');
                bodyTextHtml = `
                <div class="table-render-box">
                    <table class="render-table">
                        ${headers.length > 0 ? `<thead><tr>${ths}</tr></thead>` : ''}
                        <tbody>${trs}</tbody>
                    </table>
                </div>`;
            }

            const isEnriched = chunk.cache_status !== 'DISABLED';
            const bodyClass = isEnriched ? 'chunk-body' : 'chunk-body no-enrich';
            const enrichColHtml = isEnriched
                ? `
                <div class="chunk-enrich-col">
                    ${summaryHtml}
                    <div class="enrich-section">
                        <span class="enrich-label">Named Entities</span>
                        <div class="tag-cloud">${entitiesHtml}</div>
                    </div>
                    <div class="enrich-section">
                        <span class="enrich-label">Semantic Keywords</span>
                        <div class="tag-cloud">${keywordsHtml}</div>
                    </div>
                    ${confidenceHtml}
                    ${neighborChainHtml}
                    ${figuresHtml}
                </div>`
                : '';

            card.innerHTML = `
                <div class="chunk-header">
                    <div class="chunk-meta-left">
                        <span class="chunk-badge">#${escapeHtml(chunk.id)}</span>
                        <span class="chunk-source">📁 ${escapeHtml(chunk.metadata.source)}${chunk.metadata.sheet_name ? ` (Sheet: ${escapeHtml(chunk.metadata.sheet_name)})` : (chunk.metadata.page ? ` (Page ${chunk.metadata.page})` : '')}</span>
                    </div>
                    <div class="chunk-meta-right">
                        <span class="chunk-strategy-badge">⚙️ Strategy: ${escapeHtml(chunk.strategy || 'recursive')}</span>
                        <span class="chunk-token-count">Tokens: ${chunk.metadata.token_count}</span>
                        <button class="copy-chunk-btn" title="Copy text to clipboard">📋</button>
                    </div>
                </div>
                <div class="${bodyClass}">
                    <div class="chunk-text-col">
                        ${parentContextHtml}
                        <div class="chunk-raw-text">
                            ${bodyTextHtml}
                            <div class="text-fade-overlay"></div>
                        </div>
                        <button class="expand-text-btn">Expand raw content</button>
                        ${contextualHtml}
                    </div>
                    ${enrichColHtml}
                </div>
            `;
            
            // Hook up expand/collapse action
            const textContainer = card.querySelector('.chunk-raw-text');
            const expandBtn = card.querySelector('.expand-text-btn');
            expandBtn.addEventListener('click', () => {
                if (textContainer.classList.contains('expanded')) {
                    textContainer.classList.remove('expanded');
                    expandBtn.textContent = "Expand raw content";
                } else {
                    textContainer.classList.add('expanded');
                    expandBtn.textContent = "Collapse raw content";
                }
            });
            
            // Hook up individual chunk copy
            card.querySelector('.copy-chunk-btn').addEventListener('click', () => {
                navigator.clipboard.writeText(chunk.text);
                alert(`Copied Chunk ${idx + 1} text to clipboard!`);
            });

            chunksList.appendChild(card);
        });

        // Smooth scroll to results
        resultsPanel.scrollIntoView({ behavior: 'smooth' });
    }


    // ── Export Actions ────────────────────────────────────────────────────────
    
    // Copy all chunks as JSON
    copyJsonBtn.addEventListener('click', () => {
        if (!processedChunksData) return;
        navigator.clipboard.writeText(JSON.stringify(processedChunksData.chunks, null, 2));
        alert("Copied all chunks as formatted JSON!");
    });

    // Download JSON file
    downloadJsonBtn.addEventListener('click', () => {
        if (!processedChunksData) return;
        const blob = new Blob([JSON.stringify(processedChunksData.chunks, null, 2)], {type : 'application/json'});
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `smartchunks_${Date.now()}.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    });

    // View Raw Output modal triggers
    viewRawBtn.addEventListener('click', () => {
        if (!processedChunksData) return;
        rawJsonOutput.textContent = JSON.stringify(processedChunksData, null, 2);
        rawModal.style.display = 'flex';
    });

    // Modal Close
    const closeModal = () => { rawModal.style.display = 'none'; };
    closeModalBtn.addEventListener('click', closeModal);
    modalCloseBtn.addEventListener('click', closeModal);
    rawModal.addEventListener('click', (e) => {
        if (e.target === rawModal) closeModal();
    });

    // Modal copy
    modalCopyBtn.addEventListener('click', () => {
        navigator.clipboard.writeText(rawJsonOutput.textContent);
        alert("Copied full API output to clipboard!");
    });

    // ── Helpers ───────────────────────────────────────────────────────────────
    function escapeHtml(str) {
        if (typeof str !== 'string') return '';
        return str
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#039;');
    }

    function formatBytes(bytes, decimals = 2) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const dm = decimals < 0 ? 0 : decimals;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
    }
});
