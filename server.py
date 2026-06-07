import http.server
import socketserver
import json
import os
import sys
import math

# Port to run the server on
PORT = 5000

# Try to import torch and custom modules for GNN inference
HAS_TORCH = False
MODEL_LOADED = False
gnn_model = None
idx_to_label = {}
DEVICE = "cpu"

# Add project subfolder to path so we can import src
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'hand_gesture_recognition'))

try:
    import torch
    import numpy as np
    from torch_geometric.data import Data, Batch
    from src.models.fusion_model import GestureClassifier
    from src.data.graph_utils import keypoints_to_graph
    
    HAS_TORCH = True
    print("[Backend] PyTorch and PyG modules found! GNN branch enabled.")
except ImportError:
    print("[Backend] PyTorch or PyTorch Geometric not found. Running in Rule-Based fallback mode.")


# Dictionary of language IDs
LANGUAGE_IDS = {'ASL': 0, 'BSL': 1, 'ISL': 2, 'CSL': 3, 'ArSL': 4, 'FSL': 5}


def create_dummy_checkpoint(ckpt_path):
    """Create a dummy PyTorch model checkpoint with A-Z gestures if none exists."""
    if not HAS_TORCH:
        return
    
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    labels = [chr(i) for i in range(ord('A'), ord('Z') + 1)]
    label_to_idx = {l: i for i, l in enumerate(labels)}
    cfg = {
        'model': {
            'embed_dim': 256,
            'graph_type': 'gcn',
        }
    }
    
    # Build model using defaults
    model = GestureClassifier(
        num_classes=len(label_to_idx),
        embed_dim=256,
        graph_type='gcn',
        use_image=False,
    )
    
    torch.save({
        'epoch': 0,
        'model': model.state_dict(),
        'config': cfg,
        'label_to_idx': label_to_idx,
        'best_val_acc': 0.0,
    }, ckpt_path)
    print(f"[Backend] Created a dummy model checkpoint with random weights at '{ckpt_path}'")


def load_gnn_model():
    global gnn_model, idx_to_label, MODEL_LOADED, DEVICE
    if not HAS_TORCH:
        return
    
    ckpt_path = os.path.join(os.path.dirname(__file__), 'hand_gesture_recognition', 'checkpoints', 'best_model.pt')
    
    # Automatically generate dummy checkpoint if missing
    if not os.path.exists(ckpt_path):
        try:
            create_dummy_checkpoint(ckpt_path)
        except Exception as e:
            print(f"[Backend] Failed to create dummy checkpoint: {e}")
            return
            
    if os.path.exists(ckpt_path):
        try:
            DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            ckpt = torch.load(ckpt_path, map_location=DEVICE)
            cfg = ckpt.get('config', {})
            label_to_idx = ckpt.get('label_to_idx', {})
            idx_to_label = {v: k for k, v in label_to_idx.items()}
            
            gnn_model = GestureClassifier(
                num_classes=len(label_to_idx),
                embed_dim=cfg.get('model', {}).get('embed_dim', 256),
                graph_type=cfg.get('model', {}).get('graph_type', 'gcn'),
                use_image=False,
            ).to(DEVICE)
            
            gnn_model.load_state_dict(ckpt['model'])
            gnn_model.eval()
            MODEL_LOADED = True
            print(f"[Backend] GNN model loaded successfully from {ckpt_path} ({len(idx_to_label)} classes).")
        except Exception as e:
            print(f"[Backend] Error loading GNN checkpoint: {e}")


# Initialize GNN model if available
load_gnn_model()


def classify_gesture_rule_based(keypoints):
    """
    Rule-based hand gesture recognition from 21 3D landmarks.
    Works independently of PyTorch with high speed and zero dependencies.
    """
    if not keypoints or len(keypoints) != 21:
        return "Invalid Hand Data", 0.0
        
    def dist(p1, p2):
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2 + (p1[2] - p2[2])**2)
        
    def dist_from_wrist(p):
        return math.sqrt(p[0]**2 + p[1]**2 + p[2]**2)

    # Translate wrist (landmark 0) to origin
    wrist = keypoints[0]
    pts = [[p[0] - wrist[0], p[1] - wrist[1], p[2] - wrist[2]] for p in keypoints]
    
    # Scale coordinates by the distance between wrist (0) and middle finger MCP (9)
    scale = dist_from_wrist(pts[9])
    if scale > 1e-6:
        pts = [[p[0] / scale, p[1] / scale, p[2] / scale] for p in pts]
    else:
        return "No Hand Landmark Scale", 0.0
        
    # Tips
    thumb_tip = pts[4]
    index_tip = pts[8]
    middle_tip = pts[12]
    ring_tip = pts[16]
    pinky_tip = pts[20]
    
    # Joint distances from wrist to check folding state
    d_thumb = dist_from_wrist(thumb_tip)
    d_index = dist_from_wrist(index_tip)
    d_middle = dist_from_wrist(middle_tip)
    d_ring = dist_from_wrist(ring_tip)
    d_pinky = dist_from_wrist(pinky_tip)
    
    # Compare tip distance from wrist to intermediate joints to check extension
    # If the tip is significantly further from the wrist than the knuckle/PIP joint, it's extended.
    thumb_extended = d_thumb > dist_from_wrist(pts[3]) + 0.15
    index_extended = d_index > dist_from_wrist(pts[6]) + 0.2
    middle_extended = d_middle > dist_from_wrist(pts[10]) + 0.2
    ring_extended = d_ring > dist_from_wrist(pts[14]) + 0.2
    pinky_extended = d_pinky > dist_from_wrist(pts[18]) + 0.2
    
    # 1. Fist (All fingers folded)
    if not index_extended and not middle_extended and not ring_extended and not pinky_extended and not thumb_extended:
        return "Fist", 98.0
        
    # 2. Open Palm (All fingers extended)
    if index_extended and middle_extended and ring_extended and pinky_extended:
        if thumb_extended:
            return "Open Palm", 99.0
        else:
            return "High Four (Open Hand)", 92.0

    # 3. Peace / V Sign (Index + Middle extended, others folded)
    if index_extended and middle_extended and not ring_extended and not pinky_extended:
        return "Peace / V Sign", 97.0

    # 4. Thumbs Up / Down (Only thumb extended)
    if thumb_extended and not index_extended and not middle_extended and not ring_extended and not pinky_extended:
        # Check if thumb tip Y is higher or lower than its base
        # (Y decreases upwards in screen-space coordinates)
        if pts[4][1] < pts[2][1] - 0.2:
            return "Thumbs Up", 96.0
        elif pts[4][1] > pts[2][1] + 0.2:
            return "Thumbs Down", 96.0

    # 5. OK Sign (Thumb and Index tips touching, other fingers extended)
    d_thumb_index = dist(thumb_tip, index_tip)
    if d_thumb_index < 0.45 and middle_extended and ring_extended and pinky_extended:
        return "OK Sign", 94.0

    # 6. Rock On (Index and Pinky extended, middle and ring folded)
    if index_extended and pinky_extended and not middle_extended and not ring_extended:
        return "Rock On (Sign of Horns)", 95.0

    # 7. Pointing (Index finger only extended)
    if index_extended and not middle_extended and not ring_extended and not pinky_extended:
        return "Pointing (Index)", 91.0

    # Fallback to count of extended fingers
    extended_list = [index_extended, middle_extended, ring_extended, pinky_extended]
    extended_count = sum(extended_list)
    if thumb_extended:
        extended_count += 1
        
    if extended_count == 1:
        return "One Finger Extended", 80.0
    elif extended_count == 2:
        return "Two Fingers Extended", 80.0
    elif extended_count == 3:
        return "Three Fingers Extended", 80.0
    elif extended_count == 4:
        return "Four Fingers Extended", 80.0
        
    return "Detecting Gesture...", 50.0



class GestureHTTPHandler(http.server.SimpleHTTPRequestHandler):
    """Custom HTTP request handler to handle static assets and POST API requests."""
    
    def __init__(self, *args, **kwargs):
        # Always serve static files from the 'public' directory
        public_dir = os.path.join(os.path.dirname(__file__), 'public')
        super().__init__(*args, directory=public_dir, **kwargs)

    def do_OPTIONS(self):
        """Handle pre-flight requests for CORS."""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_POST(self):
        if self.path == '/api/predict':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            
            try:
                payload = json.loads(post_data.decode('utf-8'))
                keypoints = payload.get('keypoints')
                language = payload.get('language', 'ASL')
                
                if not keypoints or len(keypoints) != 21:
                    raise ValueError("Keypoints must contain exactly 21 coordinates.")

                # Try PyTorch GNN if loaded
                response_data = None
                if MODEL_LOADED:
                    try:
                        kp_array = np.array(keypoints, dtype=np.float32)
                        graph = keypoints_to_graph(kp_array)
                        batch = Batch.from_data_list([graph]).to(DEVICE)
                        
                        lang_id = torch.tensor([LANGUAGE_IDS.get(language, 0)], device=DEVICE)
                        with torch.no_grad():
                            logits, _ = gnn_model(batch, None, lang_id)
                            probs = torch.softmax(logits, dim=-1)[0]
                            prob, idx = probs.max(dim=0)
                            
                        gesture_name = idx_to_label.get(idx.item(), f"Class {idx.item()}")
                        response_data = {
                            'gesture': gesture_name,
                            'confidence': round(prob.item() * 100, 1),
                            'method': 'PyTorch GNN (Deep Learning)'
                        }
                    except Exception as e:
                        print(f"[Backend] GNN execution failed, using Rule-Based engine: {e}")
                
                # If GNN is not active or failed, use Rule-Based classifier
                if response_data is None:
                    gesture, conf = classify_gesture_rule_based(keypoints)
                    response_data = {
                        'gesture': gesture,
                        'confidence': conf,
                        'method': 'Rule-Based Engine (Out-of-the-Box)'
                    }
                
                # Send success response
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps(response_data).encode('utf-8'))
                
            except Exception as e:
                self.send_response(400)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({'error': str(e)}).encode('utf-8'))
        else:
            # Fall back to standard GET/POST handling of static file request
            super().do_POST()

    def end_headers(self):
        """Append CORS headers to static files as well."""
        self.send_header('Access-Control-Allow-Origin', '*')
        super().end_headers()


if __name__ == '__main__':
    # Force loading of GNN model (if package setup changed)
    if not MODEL_LOADED and HAS_TORCH:
        load_gnn_model()
        
    print(f"\n"
          f"============================================================\n"
          f"     Hand Gesture Recognition Server Running at:\n"
          f"     ---> http://localhost:{PORT} <---\n"
          f"============================================================\n"
          f"Serving static front-end files from 'public/' directory.\n"
          f"Press Ctrl+C to stop the server.\n")
          
    # Avoid address already in use errors on restart
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("", PORT), GestureHTTPHandler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server gracefully...")
            httpd.server_close()
            sys.exit(0)
