class DrawingCanvas {
    constructor() {
        this.canvas = document.getElementById('drawingCanvas');
        this.ctx = this.canvas.getContext('2d');
        this.isDrawing = false;
        this.currentBrushSize = 6;
        
        this.initializeCanvas();
        this.setupEventListeners();
        this.setupControls();
    }
    
    initializeCanvas() {
        // Set canvas background to white
        this.ctx.fillStyle = 'white';
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
        
        // Set default drawing properties
        this.ctx.lineCap = 'round';
        this.ctx.lineJoin = 'round';
        this.ctx.strokeStyle = '#000000';
        this.ctx.lineWidth = this.currentBrushSize;
    }
    
    setupEventListeners() {
        // Mouse events
        this.canvas.addEventListener('mousedown', (e) => this.startDrawing(e));
        this.canvas.addEventListener('mousemove', (e) => this.draw(e));
        this.canvas.addEventListener('mouseup', () => this.stopDrawing());
        this.canvas.addEventListener('mouseout', () => this.stopDrawing());
        
        // Touch events for mobile support
        this.canvas.addEventListener('touchstart', (e) => {
            e.preventDefault();
            const touch = e.touches[0];
            const mouseEvent = new MouseEvent('mousedown', {
                clientX: touch.clientX,
                clientY: touch.clientY
            });
            this.canvas.dispatchEvent(mouseEvent);
        });
        
        this.canvas.addEventListener('touchmove', (e) => {
            e.preventDefault();
            const touch = e.touches[0];
            const mouseEvent = new MouseEvent('mousemove', {
                clientX: touch.clientX,
                clientY: touch.clientY
            });
            this.canvas.dispatchEvent(mouseEvent);
        });
        
        this.canvas.addEventListener('touchend', (e) => {
            e.preventDefault();
            const mouseEvent = new MouseEvent('mouseup', {});
            this.canvas.dispatchEvent(mouseEvent);
        });
    }
    
    setupControls() {
        // Clear button
        const clearButton = document.getElementById('clearCanvas');
        clearButton.addEventListener('click', () => this.clearCanvas());
        
        // Brush size
        const brushSize = document.getElementById('brushSize');
        const brushSizeValue = document.getElementById('brushSizeValue');
        brushSize.addEventListener('input', (e) => {
            this.currentBrushSize = e.target.value;
            this.ctx.lineWidth = this.currentBrushSize;
            brushSizeValue.textContent = this.currentBrushSize;
        });
    }
    
    getMousePos(e) {
        const rect = this.canvas.getBoundingClientRect();
        return {
            x: e.clientX - rect.left,
            y: e.clientY - rect.top
        };
    }
    
    startDrawing(e) {
        this.isDrawing = true;
        const pos = this.getMousePos(e);
        this.ctx.beginPath();
        this.ctx.moveTo(pos.x, pos.y);
    }
    
    draw(e) {
        if (!this.isDrawing) return;
        
        const pos = this.getMousePos(e);
        this.ctx.lineTo(pos.x, pos.y);
        this.ctx.stroke();
    }
    
    stopDrawing() {
        if (this.isDrawing) {
            this.isDrawing = false;
            this.ctx.beginPath();
            
            // Convert drawing to 28x28 grayscale array when user finishes drawing
            const grayscaleArray = this.convertTo28x28Grayscale();
            
            // Send to server for prediction
            this.sendToServer(grayscaleArray);
        }
    }
    
    clearCanvas() {
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        this.initializeCanvas();
        this.resetPredictions();
    }
    
    resetPredictions() {
        // Reset all predictions to 0.0%
        for (let i = 0; i < 10; i++) {
            const element = document.getElementById(`digit-${i}`);
            const bar = document.getElementById(`bar-${i}`);
            
            if (element && bar) {
                element.textContent = '0.0%';
                bar.style.width = '0%';
                bar.style.backgroundColor = '#007bff'; // Reset to default color
            }
        }
        
        // Hide any error messages
        const errorElement = document.getElementById('error-message');
        if (errorElement) {
            errorElement.style.display = 'none';
        }
    }
    
    convertTo28x28Grayscale() {
        // Create a temporary canvas for scaling
        const tempCanvas = document.createElement('canvas');
        const tempCtx = tempCanvas.getContext('2d');
        tempCanvas.width = 28;
        tempCanvas.height = 28;
        
        // Scale the original canvas content to 28x28
        tempCtx.drawImage(this.canvas, 0, 0, this.canvas.width, this.canvas.height, 0, 0, 28, 28);
        
        // Get image data from the scaled canvas
        const imageData = tempCtx.getImageData(0, 0, 28, 28);
        const data = imageData.data;
        
        // Convert to grayscale array (784 elements)
        const grayscaleArray = [];
        
        for (let i = 0; i < data.length; i += 4) {
            // Extract RGB values (alpha channel is at i+3)
            const r = data[i];
            const g = data[i + 1];
            const b = data[i + 2];
            
            // Convert to grayscale using luminance formula
            // Standard formula: 0.299*R + 0.587*G + 0.114*B
            const grayscale = Math.round(0.299 * r + 0.587 * g + 0.114 * b);
            
            // Normalize to 0-1 range (0 = black, 1 = white)
            // For neural networks, often inverted so black = 1, white = 0
            const normalizedValue = 1 - (grayscale / 255);
            
            grayscaleArray.push(normalizedValue);
        }
        
        console.log('28x28 Grayscale Array (784 elements):', grayscaleArray);
        console.log('Array length:', grayscaleArray.length);
        
        return grayscaleArray;
    }
    
    async sendToServer(grayscaleArray) {
        const serverUrl = '/predict';
        
        try {
            // Use text/plain to avoid CORS preflight OPTIONS request
            const response = await fetch(serverUrl, {
                method: 'POST',
                headers: {
                    'Content-Type': 'text/plain',
                },
                body: JSON.stringify(grayscaleArray)
            });

            if (response.ok) {
                const predictions = await response.json();
                this.updatePredictions(predictions);
            } else {
                console.error('Server error:', response.status);
                this.showError('Server error occurred');
            }
        } catch (error) {
            console.error('Network error:', error);
            this.showError('Network error - server not available');
        }
    }
    
    updatePredictions(predictions) {
        // predictions should be an array of 10 probabilities
        for (let i = 0; i < 10; i++) {
            const probability = predictions[i] || 0;
            const percentage = (probability * 100).toFixed(1);
            const element = document.getElementById(`digit-${i}`);
            const bar = document.getElementById(`bar-${i}`);
            
            if (element && bar) {
                element.textContent = `${percentage}%`;
                bar.style.width = `${percentage}%`;
                
                // Add visual feedback for highest probability
                if (probability === Math.max(...predictions)) {
                    bar.style.backgroundColor = '#28a745';
                } else {
                    bar.style.backgroundColor = '#007bff';
                }
            }
        }
    }
    
    showError(message) {
        const errorElement = document.getElementById('error-message');
        if (errorElement) {
            errorElement.textContent = message;
            errorElement.style.display = 'block';
            setTimeout(() => {
                errorElement.style.display = 'none';
            }, 3000);
        }
    }
}

// Training-specific canvas class that doesn't auto-predict
class TrainingCanvas extends DrawingCanvas {
    constructor() {
        super();
    }
    
    // Override stopDrawing to prevent automatic prediction
    stopDrawing() {
        if (this.isDrawing) {
            this.isDrawing = false;
            this.ctx.beginPath();
            // Don't auto-send for prediction in training mode
        }
    }
    
    // Override resetPredictions since we don't have predictions in training
    resetPredictions() {
        // No predictions to reset in training mode
    }
}

class TrainingInterface {
    constructor() {
        this.currentDigit = this.generateRandomDigit();
        this.drawingCanvas = null;
        this.initializeInterface();
        this.setupEventListeners();
    }
    
    generateRandomDigit() {
        return Math.floor(Math.random() * 10);
    }
    
    initializeInterface() {
        this.updateTargetDigit();
        
        // Initialize training canvas instead of regular canvas
        this.drawingCanvas = new TrainingCanvas();
        window.drawingCanvasInstance = this.drawingCanvas;
        
        // Load initial file count
        this.loadFileCount();
    }
    
    updateTargetDigit() {
        const targetElement = document.getElementById('targetDigit');
        targetElement.textContent = this.currentDigit;
    }
    
    async loadFileCount() {
        const fileCountElement = document.getElementById('fileCountNumber');
        if (fileCountElement) {
            fileCountElement.textContent = '';
            
            try {
                const response = await fetch('/count', {
                    method: 'GET'
                });
                
                if (response.ok) {
                    const count = await response.text();
                    fileCountElement.textContent = count;
                } else {
                    console.error('Failed to load file count:', response.status);
                    fileCountElement.textContent = 'Error loading count';
                }
            } catch (error) {
                console.error('Network error while loading file count:', error);
                fileCountElement.textContent = 'Network error';
            }
        }
    }
    
    updateFileCount(count) {
        const fileCountElement = document.getElementById('fileCountNumber');
        if (fileCountElement) {
            fileCountElement.textContent = count;
        }
    }
    
    setupEventListeners() {
        const sendButton = document.getElementById('sendTrainingData');
        if (sendButton) {
            sendButton.addEventListener('click', () => this.sendTrainingData());
        }
        
        // Add keyboard event listener for space key
        document.addEventListener('keydown', (e) => {
            if (e.code === 'Space' || e.key === ' ') {
                e.preventDefault(); // Prevent page scrolling
                this.sendTrainingData();
            }
        });
    }
    
    async sendTrainingData() {
        if (!this.drawingCanvas) {
            this.showMessage('Drawing canvas not ready. Please try again.', 'error');
            return;
        }
        
        const sendButton = document.getElementById('sendTrainingData');
        sendButton.disabled = true;
        
        try {
            // Get the grayscale array from the drawing canvas
            const grayscaleArray = this.drawingCanvas.convertTo28x28Grayscale();
            
            // Prepare training data
            const trainingData = {
                digit: this.currentDigit,
                image_data: grayscaleArray
            };
            
            // Send to server
            const response = await fetch('/training', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(trainingData)
            });
            
            if (response.ok) {
                const fileCount = await response.text();
                this.updateFileCount(fileCount);
                
                // Generate next digit after successful submission
                this.nextDigit();
            } else {
                this.showMessage('Server error occurred', 'error');
            }
        } catch (error) {
            console.error('Network error:', error);
            this.showMessage('Network error - server not available', 'error');
        } finally {
            sendButton.disabled = false;
        }
    }
    
    nextDigit() {
        this.currentDigit = this.generateRandomDigit();
        this.updateTargetDigit();
        
        // Clear the canvas
        if (this.drawingCanvas) {
            this.drawingCanvas.clearCanvas();
        }
    }
    
    showMessage(message, type) {
        const statusElement = document.getElementById('statusMessage');
        if (statusElement) {
            statusElement.textContent = message;
            statusElement.className = `status-message ${type}`;
            statusElement.style.display = 'block';
            
            // Auto-hide message after 2 seconds
            setTimeout(() => {
                this.hideMessage();
            }, 2000);
        }
    }
    
    hideMessage() {
        const statusElement = document.getElementById('statusMessage');
        if (statusElement) {
            statusElement.style.display = 'none';
        }
    }
}

// Initialize the appropriate canvas based on the page
document.addEventListener('DOMContentLoaded', () => {
    // Check if we're on the training page
    if (document.getElementById('sendTrainingData')) {
        // Training page
        new TrainingInterface();
    } else {
        // Prediction page
        window.drawingCanvasInstance = new DrawingCanvas();
    }
});
