import re

with open('templates/about.html', 'r') as f:
    html = f.read()

# Remove old single-line team section if it exists
html = re.sub(r'<!-- ================= TEAM SECTION ================= -->.*?<!-- ================= FOOTER MESSAGE ================= -->', '<!-- ================= FOOTER MESSAGE ================= -->', html, flags=re.DOTALL)

team_section = """
<!-- ================= TEAM SECTION ================= -->
<section class="team-section py-5 bg-light">
    <div class="container">
        <div class="text-center mb-5">
            <h2 class="fw-bold">Project Team</h2>
            <p class="text-muted">Meet the creators behind Deepfake Analysis</p>
        </div>
        <div class="team-grid">
            <div class="team-card">
                <img src="{{ url_for('static', filename='images/member1.jpg') }}" alt="Team Member 1">
                <h4 class="mt-3">Project Partner</h4>
            </div>
            <div class="team-card lead-card">
                <img src="{{ url_for('static', filename='images/member2.jpg') }}" alt="Project Lead">
                <h4 class="mt-3">Project Lead</h4>
            </div>
            <div class="team-card">
                <img src="{{ url_for('static', filename='images/member3.jpg') }}" alt="Team Member 3">
                <h4 class="mt-3">Project Partner</h4>
            </div>
        </div>
    </div>
</section>

"""

html = html.replace('<!-- ================= FOOTER MESSAGE ================= -->', team_section + '<!-- ================= FOOTER MESSAGE ================= -->')

with open('templates/about.html', 'w') as f:
    f.write(html)
