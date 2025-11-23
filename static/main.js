const form = document.getElementById('run-form');
const statusEl = document.getElementById('status');
const planList = document.getElementById('plan-list');
const outputsEl = document.getElementById('outputs');
const summaryEl = document.getElementById('summary');
const notebookLinkEl = document.getElementById('notebook-link');

form.addEventListener('submit', async (e) => {
    e.preventDefault();
    statusEl.textContent = 'Running agents...';
    planList.innerHTML = '';
    outputsEl.innerHTML = '';
    summaryEl.textContent = '';
    notebookLinkEl.innerHTML = '';

    const data = new FormData();
    data.append('instructions', document.getElementById('instructions').value);
    if (form.file.files[0]) {
        data.append('file', form.file.files[0]);
    }

    try {
        const response = await fetch('/api/run', {
            method: 'POST',
            body: data,
        });
        const result = await response.json();
        statusEl.textContent = 'Complete';

        (result.plan || []).forEach(item => {
            const li = document.createElement('li');
            li.textContent = item;
            planList.appendChild(li);
        });

        (result.outputs || []).forEach(section => {
            const card = document.createElement('div');
            card.className = 'output-card';
            card.innerHTML = `<h3>${section.title}</h3><pre>${section.content}</pre>`;
            outputsEl.appendChild(card);
        });

        summaryEl.textContent = result.summary || '';

        if (result.notebook) {
            const link = document.createElement('a');
            link.href = `/api/notebook/${result.notebook}`;
            link.textContent = 'Download notebook';
            notebookLinkEl.appendChild(link);
        }
    } catch (err) {
        statusEl.textContent = 'Error running flow';
        console.error(err);
    }
});
