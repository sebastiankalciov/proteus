import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from PIL import Image, ImageOps
import pandas as pd
import numpy as np
import io
import os

# cnn architecture
class cnn(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(0.25),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.25),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(0.25)
        )

        self.classifier = nn.Linear(128, 3)

    def forward(self, x):
        x = self.features(x)
        x = x.mean(dim=[2, 3])  # global average pooling
        return self.classifier(x)

# utils
@st.cache_resource
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_data(batch_size):

    train_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(
            degrees=15,
            translate=(0.1, 0.1),
            scale=(0.9, 1.1)
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_set = ImageFolder("data/train", transform=train_transform)
    test_set = ImageFolder("data/test", transform=test_transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, train_set.classes


def preprocess_image(img, invert=False):
    img = img.convert("RGB")
    if invert:
        img = ImageOps.invert(img)
    img = img.resize((64, 64))

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    tensor = transform(img).unsqueeze(0)
    return tensor, img


# streamlit ui
def main():
    st.set_page_config(page_title="gesture cnn", page_icon="✊", layout="wide")

    st.title("rock-paper-scissors gesture classifier")
    st.markdown("""
    **cnn-based classifier** for hand gestures:
    **rock, paper, scissors**
    """)

    device = get_device()
    save_path = "best_gesture_model.pth"

    # initialize session state
    if "model" not in st.session_state:
        st.session_state.model = None
    if "trained" not in st.session_state:
        st.session_state.trained = False
    if "classes" not in st.session_state:
        st.session_state.classes = None

    # sidebar
    with st.sidebar:
        st.header("settings")

        mode = st.radio("mode", ["train model", "prediction playground"])

        st.divider()
        st.subheader("hyperparameters")

        epochs = st.slider("epochs", 1, 50, 10)
        lr = st.select_slider("learning rate", [0.01, 0.001, 0.0001], value=0.001)
        batch_size = st.select_slider("batch size", [16, 32, 64], value=32)

        st.caption(f"device: {device}")

        st.divider()
        st.subheader("model management")

        # download trained model
        if st.session_state.trained and st.session_state.model is not None:
            buffer = io.BytesIO()
            torch.save(st.session_state.model.state_dict(), buffer)
            buffer.seek(0)  # reset buffer position
            st.download_button(
                "download model",
                buffer,
                file_name="best_gesture_model.pth",
                mime="application/octet-stream"
            )

        # load saved model
        if os.path.exists(save_path):
            if st.button("load saved model"):
                try:
                    model = cnn().to(device)
                    model.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))
                    model.eval()
                    
                    # load classes if available
                    if os.path.exists("data/train"):
                        temp_transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])
                        temp_dataset = ImageFolder("data/train", transform=temp_transform)
                        st.session_state.classes = temp_dataset.classes
                    
                    st.session_state.model = model
                    st.session_state.trained = True
                    st.success("model loaded successfully!")
                except Exception as e:
                    st.error(f"failed to load model: {e}")

    # train mode
    if mode == "train model":
        st.subheader("🚀 training")

        if not os.path.exists("data/train") or not os.path.exists("data/test"):
            st.error("❌ dataset not found. please create `data/train` and `data/test` folders with subdirectories for each class.")
            st.info("expected structure:\n```\ndata/\n  train/\n    rock/\n    paper/\n    scissors/\n  test/\n    rock/\n    paper/\n    scissors/\n```")
            return

        if st.button("start training", type="primary"):
            with st.spinner("loading data..."):
                train_loader, test_loader, classes = load_data(batch_size)
                
            st.session_state.classes = classes
            
            st.info(f"📊 training on {len(train_loader.dataset)} samples, validating on {len(test_loader.dataset)} samples")
            st.info(f"🏷️ classes: {', '.join(classes)}")

            model = cnn().to(device)
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
            criterion = nn.CrossEntropyLoss()

            progress = st.progress(0)
            status_text = st.empty()
            chart = st.empty()

            train_loss_hist = []
            val_loss_hist = []
            val_acc_hist = []

            best_val = float("inf")
            patience = 5
            counter = 0

            for epoch in range(epochs):
                # train
                model.train()
                total_loss = 0

                for x, y in train_loader:
                    x, y = x.to(device), y.to(device)

                    optimizer.zero_grad()
                    out = model(x)
                    loss = criterion(out, y)
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()

                train_loss = total_loss / len(train_loader)
                train_loss_hist.append(train_loss)

                # validate
                model.eval()
                val_loss = 0
                correct = 0
                total = 0

                with torch.no_grad():
                    for x, y in test_loader:
                        x, y = x.to(device), y.to(device)
                        out = model(x)
                        loss = criterion(out, y)
                        val_loss += loss.item()

                        preds = torch.argmax(out, 1)
                        correct += (preds == y).sum().item()
                        total += y.size(0)

                val_loss /= len(test_loader)
                val_loss_hist.append(val_loss)
                acc = correct / total
                val_acc_hist.append(acc)

                # save best model and early stopping
                if val_loss < best_val:
                    best_val = val_loss
                    torch.save(model.state_dict(), save_path)
                    counter = 0
                else:
                    counter += 1
                    if counter >= patience:
                        st.info(f"early stopping at epoch {epoch+1}")
                        break

                # update ui
                df = pd.DataFrame({
                    "train loss": train_loss_hist,
                    "val loss": val_loss_hist
                })
                chart.line_chart(df)

                progress.progress((epoch + 1) / epochs)
                status_text.write(f"**epoch {epoch+1}/{epochs}** | train loss: {train_loss:.4f} | val loss: {val_loss:.4f} | val acc: {acc*100:.2f}%")

            # load best model
            model.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))
            model.eval()
            st.session_state.model = model
            st.session_state.trained = True
            
            st.success(f"✅ training complete! best validation accuracy: {max(val_acc_hist)*100:.2f}%")
            
            # show final metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("final train loss", f"{train_loss_hist[-1]:.4f}")
            with col2:
                st.metric("final val loss", f"{val_loss_hist[-1]:.4f}")
            with col3:
                st.metric("best val accuracy", f"{max(val_acc_hist)*100:.2f}%")

    # predict mode
    else:
        st.subheader("prediction playground")

        if not st.session_state.trained or st.session_state.model is None:
            st.warning("please train or load a model first.")
            return

        if st.session_state.classes is None:
            st.error("class names not loaded, please retrain or ensure data/train folder exists")
            return

        model = st.session_state.model
        model.eval()

        uploaded = st.file_uploader("upload an image", type=["png", "jpg", "jpeg"])
        invert = st.checkbox("invert colors", value=False, help="enable if your image has inverted colors")

        if uploaded:
            try:
                img = Image.open(uploaded)
                tensor, debug = preprocess_image(img, invert)
                tensor = tensor.to(device)

                with torch.no_grad():
                    out = model(tensor)
                    probs = torch.softmax(out, 1)[0].cpu().numpy()

                pred_idx = int(np.argmax(probs))
                pred_label = st.session_state.classes[pred_idx]
                confidence = probs[pred_idx]

                col1, col2 = st.columns([1, 2])

                with col1:
                    st.image(debug, caption="preprocessed input", width=200)

                with col2:
                    st.metric("prediction", pred_label, f"{confidence*100:.1f}% confidence")
                    
                    # probability distribution
                    df = pd.DataFrame({
                        "probability": probs
                    }, index=st.session_state.classes)
                    st.bar_chart(df)
                    
                    # show raw probabilities
                    with st.expander("view detailed probabilities"):
                        for i, class_name in enumerate(st.session_state.classes):
                            st.write(f"**{class_name}**: {probs[i]*100:.2f}%")
                            
            except Exception as e:
                st.error(f"error processing image {e}")

if __name__ == "__main__":
    main()