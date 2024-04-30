function replaceText(element) {
    if (element.nodeType === Node.TEXT_NODE) {
        let text = element.textContent;
        if (text.includes("Selected:")) {
            text = text.replace("Selected:", "Ihre Auswahl:");
        }
        if (text.includes("You")) {
            text = text.replace(/You/g, "Ich");
        }
        element.textContent = text;
    } else {
        for (const child of element.childNodes) {
            replaceText(child);
        }
    }
}

function observeChanges() {
    const observer = new MutationObserver((mutations) => {
        mutations.forEach((mutation) => {
            if (mutation.addedNodes.length) {
                mutation.addedNodes.forEach((addedNode) => {
                    replaceText(addedNode);
                });
            }
        });
    });

    const config = { childList: true, subtree: true };

    observer.observe(document.body, config);
}


function createCustomAlert() {
    const alertBox = document.createElement('div');
    alertBox.setAttribute('id', 'customAlertBox');
    alertBox.innerHTML = `
        <div id="alertContent">
            <p>Willkommen bei LexyAI! <br>
            Ich bin ein KI-gestützter Chatbot, welcher mit OpenAIs KI-Systemen interagiert. <br>
            Bitte beachte also die einschlägigen Anforderungen der DS-GVO. <br>
            OK, um fortzufahren.</p>
            <button onclick="closeCustomAlert()">OK</button>
        </div>
    `;
    document.body.appendChild(alertBox);
}


function closeCustomAlert() {
    const alertBox = document.getElementById('customAlertBox');
    document.body.removeChild(alertBox);
}

document.addEventListener('DOMContentLoaded', createCustomAlert);
document.addEventListener('DOMContentLoaded', observeChanges);
