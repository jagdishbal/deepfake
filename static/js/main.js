// DeepFake Analysis Main Client Script
document.addEventListener('DOMContentLoaded', function () {
    // Auto-dismiss alerts after 6 seconds
    const alerts = document.querySelectorAll('.alert-dismissible');
    alerts.forEach(function (alert) {
        setTimeout(function () {
            try {
                const bsAlert = bootstrap.Alert.getOrCreateInstance(alert);
                if (bsAlert) {
                    bsAlert.close();
                }
            } catch (e) {
                alert.style.display = 'none';
            }
        }, 6000);
    });

    // Provide immediate feedback on forms when submitting
    const forms = document.querySelectorAll('form[enctype="multipart/form-data"]');
    forms.forEach(function (form) {
        form.addEventListener('submit', function (e) {
            const submitBtn = form.querySelector('button[type="submit"]');
            if (submitBtn && !submitBtn.disabled) {
                const originalHtml = submitBtn.innerHTML;
                submitBtn.disabled = true;
                submitBtn.innerHTML = '<span class="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span>Analyzing... Please wait';
                // If form submission takes long or fails validation, restore after 30s timeout
                setTimeout(function () {
                    submitBtn.disabled = false;
                    submitBtn.innerHTML = originalHtml;
                }, 30000);
            }
        });
    });
});
