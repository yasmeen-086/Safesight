import { useState, useRef, useCallback } from 'react';
import ViolationsDialog from '../components/ViolationsDialog';
import './Upload.css';

const PROCESS_STEPS = [
    'Enhancing frames',
    'Running YOLO detection',
    'Computing homography transform',
    'Mapping violations',
];

export default function Upload() {
    const [file, setFile] = useState(null);
    const [dragOver, setDragOver] = useState(false);
    const [processing, setProcessing] = useState(false);
    const [currentStep, setCurrentStep] = useState(-1);
    const [progress, setProgress] = useState(0);
    const [done, setDone] = useState(false);
    const [showDialog, setShowDialog] = useState(false);
    const inputRef = useRef(null);

    const handleFile = useCallback((f) => {
        if (f && (f.type.startsWith('video/') || /\.(mp4|avi|mov|mkv)$/i.test(f.name))) {
            setFile(f);
            setProcessing(false);
            setCurrentStep(-1);
            setProgress(0);
            setDone(false);
        }
    }, []);

    const onDrop = (e) => {
        e.preventDefault();
        setDragOver(false);
        if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
    };

    const simulateProcessing = () => {
        setProcessing(true);
        setDone(false);
        setProgress(0);
        setCurrentStep(0);

        let step = 0;
        const interval = setInterval(() => {
            step++;
            const pct = Math.min((step / PROCESS_STEPS.length) * 100, 100);
            setProgress(pct);

            if (step < PROCESS_STEPS.length) {
                setCurrentStep(step);
            } else {
                clearInterval(interval);
                setCurrentStep(PROCESS_STEPS.length);
                setProcessing(false);
                setDone(true);
            }
        }, 1400);
    };

    const resetUpload = () => {
        setFile(null);
        setProcessing(false);
        setCurrentStep(-1);
        setProgress(0);
        setDone(false);
    };

    const formatSize = (bytes) => {
        if (bytes < 1024) return bytes + ' B';
        if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
        return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
    };

    return (
        <div className="upload-page">
            <div className="upload-header">
                <div className="container">
                    <p className="section-label">Analysis</p>
                    <h1 className="section-title">Upload footage</h1>
                    <p className="section-subtitle">
                        Upload your CCTV video file to run the safety analysis pipeline.
                        Supported formats include MP4, AVI, and MOV.
                    </p>
                </div>
            </div>

            <div className="upload-body">
                <div className="container">
                    {/* Drop Zone */}
                    {!file && (
                        <div
                            className={`drop-zone ${dragOver ? 'drag-over' : ''}`}
                            onClick={() => inputRef.current?.click()}
                            onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
                            onDragLeave={() => setDragOver(false)}
                            onDrop={onDrop}
                        >
                            <div className="drop-zone-icon">↑</div>
                            <h3>Drop your video file here</h3>
                            <p>or click to browse from your computer</p>
                            <div className="file-types">
                                <span className="file-type-badge">.mp4</span>
                                <span className="file-type-badge">.avi</span>
                                <span className="file-type-badge">.mov</span>
                                <span className="file-type-badge">.mkv</span>
                            </div>
                            <input
                                ref={inputRef}
                                type="file"
                                accept="video/*"
                                style={{ display: 'none' }}
                                onChange={(e) => e.target.files.length && handleFile(e.target.files[0])}
                            />
                        </div>
                    )}

                    {/* File Selected */}
                    {file && (
                        <>
                            <div className="file-info">
                                <div className="file-meta">
                                    <div className="file-icon">▶</div>
                                    <div className="file-details">
                                        <h4>{file.name}</h4>
                                        <span>{formatSize(file.size)}</span>
                                    </div>
                                </div>
                                {!processing && !done && (
                                    <button className="remove-file" onClick={resetUpload}>✕</button>
                                )}
                            </div>

                            {/* Start / Processing */}
                            {!processing && !done && currentStep === -1 && (
                                <div style={{ marginTop: 24 }}>
                                    <button className="btn btn-primary" onClick={simulateProcessing}>
                                        Run Analysis →
                                    </button>
                                </div>
                            )}

                            {/* Progress */}
                            {(processing || done) && (
                                <div className="progress-section">
                                    <div className="progress-header">
                                        <span>{done ? 'Complete' : 'Processing...'}</span>
                                        <span>{Math.round(progress)}%</span>
                                    </div>
                                    <div className="progress-bar-container">
                                        <div
                                            className="progress-bar-fill"
                                            style={{ width: `${progress}%` }}
                                        />
                                    </div>
                                    <div className="processing-steps">
                                        {PROCESS_STEPS.map((step, i) => {
                                            let status = '';
                                            if (i < currentStep) status = 'done';
                                            else if (i === currentStep && processing) status = 'active';
                                            else if (i === currentStep && done) status = 'done';

                                            return (
                                                <div className={`proc-step ${status}`} key={i}>
                                                    {status === 'active' ? (
                                                        <div className="spinner" />
                                                    ) : (
                                                        <div className="proc-step-icon">
                                                            {status === 'done' ? '✓' : (i + 1)}
                                                        </div>
                                                    )}
                                                    <span>{step}{status === 'active' ? '…' : ''}</span>
                                                </div>
                                            );
                                        })}
                                    </div>
                                </div>
                            )}

                            {/* Results */}
                            {done && (
                                <div className="results-section">
                                    <h3>Analysis complete</h3>
                                    <p>
                                        7 safety violations detected across 4 zones in {file.name}
                                    </p>
                                    <button
                                        className="btn btn-primary"
                                        onClick={() => setShowDialog(true)}
                                    >
                                        View Results
                                    </button>
                                    <button
                                        className="btn btn-outline"
                                        onClick={resetUpload}
                                        style={{ marginLeft: 12 }}
                                    >
                                        Upload Another
                                    </button>
                                </div>
                            )}
                        </>
                    )}
                </div>
            </div>

            {/* Violations Dialog */}
            {showDialog && (
                <ViolationsDialog
                    filename={file?.name || 'video.mp4'}
                    onClose={() => setShowDialog(false)}
                />
            )}
        </div>
    );
}
