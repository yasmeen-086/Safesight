import { Link } from 'react-router-dom';
import './Home.css';

const features = [
    {
        icon: '⊡',
        title: 'Real-Time Detection',
        desc: 'YOLO-powered detection identifies workers, equipment, and potential hazards frame-by-frame from live CCTV feeds.',
    },
    {
        icon: '◫',
        title: "Bird's-Eye Mapping",
        desc: 'Projective transforms convert oblique camera angles into accurate top-down ground-plane maps using homography matrices.',
    },
    {
        icon: '◈',
        title: 'Adaptive Enhancement',
        desc: 'Automatic frame correction — CLAHE, gamma adjustment, sharpening — triggered by real-time scene analysis.',
    },
];

const steps = [
    {
        title: 'Upload Footage',
        desc: 'Drop in your CCTV video files for analysis.',
    },
    {
        title: 'Frame Enhancement',
        desc: 'Each frame is adaptively enhanced based on brightness, contrast, and blur metrics.',
    },
    {
        title: 'Object Detection',
        desc: 'YOLO identifies workers, helmets, vests, and restricted equipment in every frame.',
    },
    {
        title: 'Perspective Transform',
        desc: 'A 3×3 homography matrix maps detections from camera space to the real-world ground plane.',
    },
    {
        title: 'Violation Alerts',
        desc: 'Zone boundaries are checked against mapped positions. Violations generate timestamped reports.',
    },
];

const techStack = [
    'Python',
    'OpenCV',
    'YOLOv8',
    'NumPy',
    'React',
    'Homography',
];

export default function Home() {
    return (
        <>
            {/* ── Hero ── */}
            <section className="hero section" id="hero">
                <div className="container">
                    <div className="hero-content">
                        <div className="hero-label">
                            <span className="dot" />
                            Computer Vision Project — UCS532
                        </div>
                        <h1>
                            Intelligent safety
                            <br />
                            monitoring for
                            <br />
                            <span className="highlight">industrial environments.</span>
                        </h1>
                        <p className="hero-desc">
                            Safesight transforms standard CCTV footage into actionable safety
                            intelligence — detecting workers, mapping positions to real-world
                            coordinates, and flagging zone violations in real time.
                        </p>
                        <div className="hero-actions">
                            <Link to="/upload" className="btn btn-primary">
                                Try It Out →
                            </Link>
                            <a href="#how-it-works" className="btn btn-outline">
                                How It Works
                            </a>
                        </div>
                    </div>
                </div>
            </section>

            {/* ── Features ── */}
            <section className="features section" id="features">
                <div className="container">
                    <p className="section-label">Capabilities</p>
                    <h2 className="section-title">What Safesight does</h2>
                    <p className="section-subtitle">
                        Three core modules working together to turn raw camera feeds into
                        safety insights.
                    </p>
                    <div className="features-grid">
                        {features.map((f, i) => (
                            <div className="card feature-card" key={i}>
                                <div className="feature-icon">{f.icon}</div>
                                <h3>{f.title}</h3>
                                <p>{f.desc}</p>
                            </div>
                        ))}
                    </div>
                </div>
            </section>

            {/* ── How It Works ── */}
            <section className="section" id="how-it-works">
                <div className="container">
                    <p className="section-label">Pipeline</p>
                    <h2 className="section-title">How it works</h2>
                    <p className="section-subtitle">
                        A five-stage computer vision pipeline from raw footage to actionable
                        safety reports.
                    </p>
                    <div className="pipeline-steps">
                        {steps.map((s, i) => (
                            <div className="pipeline-step" key={i}>
                                <span className="step-number">0{i + 1}</span>
                                <div className="step-content">
                                    <h3>{s.title}</h3>
                                    <p>{s.desc}</p>
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            </section>

            {/* ── Tech Stack ── */}
            <section className="tech-stack section" id="tech-stack">
                <div className="container">
                    <p className="section-label">Built With</p>
                    <h2 className="section-title">Tech stack</h2>
                    <div className="tech-grid">
                        {techStack.map((t) => (
                            <span className="tech-badge" key={t}>
                                <span className="tech-dot" />
                                {t}
                            </span>
                        ))}
                    </div>
                </div>
            </section>

            {/* ── Footer ── */}
            <footer className="footer">
                <div className="container">
                    <p>Safesight — UCS532 Computer Vision Project © 2026</p>
                </div>
            </footer>
        </>
    );
}
