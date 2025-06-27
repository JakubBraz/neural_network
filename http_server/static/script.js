class DrawingCanvas {
    constructor() {
        this.canvas = document.getElementById('drawingCanvas');
        this.ctx = this.canvas.getContext('2d');
        this.isDrawing = false;
        this.currentBrushSize = 2;
        
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

// Initialize the drawing canvas when the page loads
document.addEventListener('DOMContentLoaded', () => {
    new DrawingCanvas();
});
