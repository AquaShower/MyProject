import os
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler

# ========== 配置 =============
TRAIN_NUM = 20000
TEST_NUM = 10000
IMG_SIZE = 28
BATCH_SIZE = 32
EPOCHS = 50
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CLUSTER_VIS_PATH = 'cluster_vis.png'

if torch.cuda.is_available():
    AMP_ENABLED = True
else:
    from contextlib import nullcontext
    autocast = nullcontext
    GradScaler = None
    AMP_ENABLED = False

# ========== 特征提取 =============
def extract_features(img_arr):
    feats = []
    feats.append(np.mean(img_arr))
    feats.append(np.var(img_arr))
    hist, _ = np.histogram(img_arr, bins=32, range=(0, 255))
    feats.extend(hist.tolist())
    fft = np.fft.fft2(img_arr)
    fft_shift = np.fft.fftshift(fft)
    mag = np.abs(fft_shift)
    center = mag[IMG_SIZE//4:3*IMG_SIZE//4, IMG_SIZE//4:3*IMG_SIZE//4]
    low_freq = np.mean(center)
    high_freq = np.mean(mag) - low_freq
    feats.append(low_freq)
    feats.append(high_freq)
    return np.array(feats)

def get_pca_features(feature_matrix, n_components=8):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(feature_matrix)
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    return X_pca

# ========== 聚类 =============
def cluster_images(features, method='kmeans', max_clusters=6):
    best_score = -1
    best_labels = None
    best_n = 2
    for n in range(2, max_clusters+1):
        try:
            if method == 'kmeans':
                model = KMeans(n_clusters=n, random_state=42)
                labels = model.fit_predict(features)
            else:
                model = GaussianMixture(n_components=n, random_state=42)
                labels = model.fit_predict(features)
            score = silhouette_score(features, labels)
            if score > best_score:
                best_score = score
                best_labels = labels
                best_n = n
        except Exception as e:
            print(f'聚类数 {n} 时聚类失败: {e}')
            continue
    if best_labels is None:
        raise RuntimeError('聚类失败，无法获得有效的 labels。')
    print(f'最佳聚类数: {best_n}, 轮廓系数: {best_score:.4f}')
    return best_labels, best_n

def visualize_clusters(features, labels, save_path=CLUSTER_VIS_PATH):
    pca = PCA(n_components=2)
    X_vis = pca.fit_transform(features)
    plt.figure(figsize=(8,6))
    for l in np.unique(labels):
        plt.scatter(X_vis[labels==l,0], X_vis[labels==l,1], label=f'Cluster {l}')
    plt.legend()
    plt.title('Image Clusters Visualization (PCA)')
    plt.savefig(save_path)
    plt.close()

# ========== PyTorch Dataset =============
class StegDataset(Dataset):
    def __init__(self, carrier_imgs, secret_img, labels=None):
        self.carrier_imgs = carrier_imgs
        self.secret_img = secret_img
        self.labels = labels
    def __len__(self):
        return len(self.carrier_imgs)
    def __getitem__(self, idx):
        carrier = self.carrier_imgs[idx]
        secret = self.secret_img
        if self.labels is not None:
            return torch.FloatTensor(carrier), torch.FloatTensor(secret), torch.tensor(self.labels[idx], dtype=torch.long)
        else:
            return torch.FloatTensor(carrier), torch.FloatTensor(secret)

# ========== Encoder/Decoder 架构 =============
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + x)

class AttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        attn = self.sigmoid(self.conv(x))
        return x * attn

class Encoder(nn.Module):
    def __init__(self, in_channels=4, out_channels=3, base_channels=128, num_res=8, num_attn=0, downsample=True):
        super().__init__()
        layers = [nn.Conv2d(in_channels, base_channels, 3, padding=1), nn.ReLU()]
        for _ in range(num_res):
            layers.append(ResidualBlock(base_channels))
        for _ in range(num_attn):
            layers.append(AttentionBlock(base_channels))
        if downsample:
            layers.append(nn.Conv2d(base_channels, base_channels, 3, stride=2, padding=1))
            layers.append(nn.ReLU())
            layers.append(nn.ConvTranspose2d(base_channels, base_channels, 3, stride=2, padding=1, output_padding=1))
            layers.append(nn.ReLU())
        layers.append(nn.Conv2d(base_channels, out_channels, 3, padding=1))
        self.net = nn.Sequential(*layers)
    def forward(self, carrier, secret):
        x = torch.cat([carrier, secret], dim=1)
        return self.net(x)

class Decoder(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, base_channels=128, num_res=8, num_attn=0, downsample=True):
        super().__init__()
        layers = [nn.Conv2d(in_channels, base_channels, 3, padding=1), nn.ReLU()]
        for _ in range(num_res):
            layers.append(ResidualBlock(base_channels))
        for _ in range(num_attn):
            layers.append(AttentionBlock(base_channels))
        if downsample:
            layers.append(nn.Conv2d(base_channels, base_channels, 3, stride=2, padding=1))
            layers.append(nn.ReLU())
            layers.append(nn.ConvTranspose2d(base_channels, base_channels, 3, stride=2, padding=1, output_padding=1))
            layers.append(nn.ReLU())
        layers.append(nn.Conv2d(base_channels, out_channels, 3, padding=1))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return torch.sigmoid(self.net(x))

# ========== 分类器 =============
class ClusterClassifier(nn.Module):
    def __init__(self, num_classes, base_channels=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, base_channels, 3, padding=1), nn.ReLU(),
            nn.Conv2d(base_channels, base_channels, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(base_channels, num_classes)
    def forward(self, x):
        feat = self.conv(x).view(x.size(0), -1)
        return self.fc(feat)

# ========== 训练与评测 =============
def psnr_ssim(img1, img2):
    img1 = img1.astype(np.uint8)
    img2 = img2.astype(np.uint8)
    
    psnr = peak_signal_noise_ratio(img1, img2, data_range=255)
    
    # 确保 win_size 不超过图像的最小边长
    min_size = min(img1.shape[:2])
    
    # 如果图像太小，直接返回默认值
    if min_size < 3:
        return psnr, 0.0
    
    win_size = min(7, min_size if min_size % 2 == 1 else min_size - 1)
    if win_size < 3:
        win_size = 3
    
    try:
        # 检查是否为多通道图像
        if len(img1.shape) == 3 and img1.shape[2] in [1, 3]:
            # 多通道图像，指定 channel_axis
            ssim = structural_similarity(img1, img2, data_range=255, win_size=win_size, channel_axis=2)
        else:
            # 单通道图像
            ssim = structural_similarity(img1, img2, data_range=255, win_size=win_size)
    except Exception:
        # 静默处理异常，不输出调试信息
        ssim = 0.0
    
    return psnr, ssim

# 尝试引入 pytorch-ssim，如果没有则只用 MSE
try:
    import pytorch_ssim
    SSIM_AVAILABLE = True
    def ssim_loss(x, y):
        return 1 - pytorch_ssim.ssim(x, y)
except ImportError:
    SSIM_AVAILABLE = False
    def ssim_loss(x, y):
        return 0.0

def train_encoder_decoder(encoder, decoder, dataloader, epochs=EPOCHS, desc=""):
    encoder, decoder = encoder.to(DEVICE), decoder.to(DEVICE)
    opt = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=5)
    mse_loss = nn.MSELoss()
    scaler = GradScaler() if AMP_ENABLED else None  # 只在AMP_ENABLED时初始化

    best_loss = float('inf')
    patience_counter = 0
    patience = 10

    for epoch in tqdm(range(epochs), desc=desc):
        encoder.train()
        decoder.train()
        epoch_loss = 0.0
        batch_count = 0

        batch_iter = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        for carrier, secret in batch_iter:
            carrier = carrier.unsqueeze(1).to(DEVICE) / 255.0
            secret = secret.unsqueeze(1).to(DEVICE) / 255.0
            carrier_rgb = carrier.repeat(1,3,1,1)
            secret_gray = secret

            if AMP_ENABLED:
                with autocast('cuda'):
                    stego = encoder(carrier_rgb, secret_gray)
                    decoded = decoder(stego)
                    # 添加载体图片保持损失
                    carrier_loss = mse_loss(stego, carrier_rgb)
                    if SSIM_AVAILABLE:
                        secret_loss = 0.3 * mse_loss(decoded, secret_gray) + 0.7 * ssim_loss(decoded, secret_gray)
                    else:
                        secret_loss = mse_loss(decoded, secret_gray)
                    # 平衡载体图片保持和秘密图片恢复
                    loss = 0.4 * carrier_loss + 0.6 * secret_loss
                opt.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                stego = encoder(carrier_rgb, secret_gray)
                decoded = decoder(stego)
                # 添加载体图片保持损失
                carrier_loss = mse_loss(stego, carrier_rgb)
                if SSIM_AVAILABLE:
                    secret_loss = 0.3 * mse_loss(decoded, secret_gray) + 0.7 * ssim_loss(decoded, secret_gray)
                else:
                    secret_loss = mse_loss(decoded, secret_gray)
                # 平衡载体图片保持和秘密图片恢复
                loss = 0.4 * carrier_loss + 0.6 * secret_loss
                opt.zero_grad()
                loss.backward()
                opt.step()

            epoch_loss += loss.item()
            batch_count += 1
            batch_iter.set_postfix(loss=loss.item())

        avg_loss = epoch_loss / batch_count
        scheduler.step(avg_loss)

        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    return encoder, decoder

def train_classifier(classifier, dataloader, epochs=EPOCHS):
    classifier = classifier.to(DEVICE)
    opt = optim.Adam(classifier.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    
    for epoch in tqdm(range(epochs), desc="训练分类器"):
        classifier.train()
        epoch_loss = 0.0
        batch_count = 0
        
        batch_iter = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        for carrier, secret, label in batch_iter:
            carrier = carrier.unsqueeze(1).to(DEVICE) / 255.0
            carrier_rgb = carrier.repeat(1,3,1,1)
            label = label.to(DEVICE)
            out = classifier(carrier_rgb)
            loss = loss_fn(out, label)
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            epoch_loss += loss.item()
            batch_count += 1
            batch_iter.set_postfix(loss=loss.item())
        
        avg_loss = epoch_loss / batch_count
        print(f"分类器 Epoch {epoch+1}/{epochs}, 平均Loss: {avg_loss:.4f}")
    
    return classifier

# ========== 主流程 =============
def main():
    # 1. 读取数据
    train_csv = 'train_data.csv'
    test_csv = 'test_data.csv'
    secret_path = 'secret.png'
    out_dir = 'output'
    os.makedirs(out_dir, exist_ok=True)
    train_data = pd.read_csv(train_csv, header=None).values[:TRAIN_NUM]
    test_data = pd.read_csv(test_csv, header=None).values[:TEST_NUM]
    # 2. 预处理秘密图片
    secret_img = Image.open(secret_path).convert('L').resize((IMG_SIZE, IMG_SIZE), Image.Resampling.LANCZOS)
    secret_arr = np.array(secret_img)
    # 3. 提取特征并聚类
    print('提取特征并聚类...')
    all_imgs = np.concatenate([train_data, test_data], axis=0)
    carriers = []
    features = []
    for row in tqdm(all_imgs, desc='提取特征'):
        if row.size == 785:
            img = row[1:].reshape(IMG_SIZE, IMG_SIZE).astype(np.uint8)
        else:
            img = row.reshape(IMG_SIZE, IMG_SIZE).astype(np.uint8)
        carriers.append(img)
        features.append(extract_features(img))
    features = np.array(features)
    # PCA降维
    pca_feats = get_pca_features(features, n_components=8)
    feats_for_cluster = np.concatenate([features, pca_feats], axis=1)
    # 聚类
    labels, n_clusters = cluster_images(feats_for_cluster, method='kmeans', max_clusters=6)
    visualize_clusters(feats_for_cluster, labels, save_path=CLUSTER_VIS_PATH)
    print(f'聚类可视化已保存到 {CLUSTER_VIS_PATH}')
    # 4. 按类别分配Encoder/Decoder参数
    cluster_params = []
    for c in range(n_clusters):
        # 高频类别: num_res=8, num_attn=0, downsample=True
        # 低频类别: num_res=4, num_attn=6, downsample=False
        # 其余: num_res=6, num_attn=3, downsample=True
        idxs = np.where(labels == c)[0]
        mean_high = np.mean(features[idxs, -2])  # high_freq
        mean_low = np.mean(features[idxs, -1])   # low_freq
        if mean_high > mean_low:
            cluster_params.append({'num_res':8, 'num_attn':0, 'downsample':True})
        elif mean_low > mean_high:
            cluster_params.append({'num_res':4, 'num_attn':6, 'downsample':False})
        else:
            cluster_params.append({'num_res':6, 'num_attn':3, 'downsample':True})
    # 5. 训练每类Encoder/Decoder
    print('训练各类Encoder/Decoder...')
    encoders, decoders = [], []
    for c in tqdm(range(n_clusters), desc="类别训练进度"):
        idxs = np.where(labels[:TRAIN_NUM] == c)[0]
        carrier_imgs = [carriers[i] for i in idxs]
        secret_imgs = [secret_arr for _ in idxs]
        dataset = StegDataset(carrier_imgs, secret_arr)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
        encoder = Encoder(in_channels=4, out_channels=3, base_channels=128, **cluster_params[c])
        decoder = Decoder(in_channels=3, out_channels=1, base_channels=128, **cluster_params[c])
        encoder, decoder = train_encoder_decoder(encoder, decoder, dataloader, desc=f"类别{c}")
        encoders.append(encoder)
        decoders.append(decoder)
    # 6. 训练分类器
    print('训练分类器...')
    train_labels = labels[:TRAIN_NUM]
    train_carriers = [carriers[i] for i in range(TRAIN_NUM)]
    dataset = StegDataset(train_carriers, secret_arr, train_labels)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    classifier = ClusterClassifier(num_classes=n_clusters, base_channels=128)
    classifier = train_classifier(classifier, dataloader)
    # 7. 性能评测
    print('评测性能...')
    test_labels = labels[TRAIN_NUM:TRAIN_NUM+TEST_NUM]
    test_carriers = [carriers[i] for i in range(TRAIN_NUM, TRAIN_NUM+TEST_NUM)]
    ssim_cover, psnr_cover = [], []
    ssim_secret, psnr_secret = [], []
    ssim_original_vs_stego, psnr_original_vs_stego = [], []  # 新增：原始载体vs隐写载体
    
    for i, (img, label) in enumerate(zip(test_carriers, test_labels)):
        carrier = torch.FloatTensor(img).unsqueeze(0).unsqueeze(0).to(DEVICE) / 255.0
        carrier_rgb = carrier.repeat(1,3,1,1)
        secret = torch.FloatTensor(secret_arr).unsqueeze(0).unsqueeze(0).to(DEVICE) / 255.0
        encoder = encoders[label].to(DEVICE)
        decoder = decoders[label].to(DEVICE)
        with torch.no_grad():
            stego = encoder(carrier_rgb, secret)
            decoded = decoder(stego)
        stego_img = (stego.squeeze().cpu().numpy().transpose(1,2,0)*255).clip(0,255).astype(np.uint8)
        orig_img = (carrier_rgb.squeeze().cpu().numpy().transpose(1,2,0)*255).clip(0,255).astype(np.uint8)
        
        # 添加调试信息（前几个样本）
        if i < 3:
            print(f"Sample {i}: stego range [{stego.min():.3f}, {stego.max():.3f}], carrier range [{carrier_rgb.min():.3f}, {carrier_rgb.max():.3f}]")
        
        # 原始载体图片与隐写后载体图片的对比
        psnr_orig_stego, ssim_orig_stego = psnr_ssim(orig_img, stego_img)
        ssim_original_vs_stego.append(ssim_orig_stego)
        psnr_original_vs_stego.append(psnr_orig_stego)
        
        # 原有的评测（这些可能有问题，暂时保留）
        psnr_c, ssim_c = psnr_ssim(orig_img, stego_img)
        ssim_cover.append(ssim_c)
        psnr_cover.append(psnr_c)
        decoded_img = (decoded.squeeze().cpu().numpy()*255).clip(0,255).astype(np.uint8)
        psnr_s, ssim_s = psnr_ssim(secret_arr, decoded_img)
        ssim_secret.append(ssim_s)
        psnr_secret.append(psnr_s)
    
    print(f'原始载体图片与隐写后载体图片PSNR均值: {np.mean(psnr_original_vs_stego):.2f}, SSIM均值: {np.mean(ssim_original_vs_stego):.4f}')
    print(f'载体图片编码前后PSNR均值: {np.mean(psnr_cover):.2f}, SSIM均值: {np.mean(ssim_cover):.4f}')
    print(f'隐藏图片藏入前后(解码后)PSNR均值: {np.mean(psnr_secret):.2f}, SSIM均值: {np.mean(ssim_secret):.4f}')
    
    # 保存性能指标到文件
    with open(os.path.join(out_dir, 'performance_results.txt'), 'w', encoding='utf-8') as f:
        f.write(f'=== 隐写性能评测结果 ===\n\n')
        f.write(f'1. 原始载体图片与隐写后载体图片对比:\n')
        f.write(f'   PSNR均值: {np.mean(psnr_original_vs_stego):.2f}, 标准差: {np.std(psnr_original_vs_stego):.2f}\n')
        f.write(f'   SSIM均值: {np.mean(ssim_original_vs_stego):.4f}, 标准差: {np.std(ssim_original_vs_stego):.4f}\n\n')
        f.write(f'2. 载体图片编码前后对比:\n')
        f.write(f'   PSNR均值: {np.mean(psnr_cover):.2f}, 标准差: {np.std(psnr_cover):.2f}\n')
        f.write(f'   SSIM均值: {np.mean(ssim_cover):.4f}, 标准差: {np.std(ssim_cover):.4f}\n\n')
        f.write(f'3. 隐藏图片藏入前后(解码后)对比:\n')
        f.write(f'   PSNR均值: {np.mean(psnr_secret):.2f}, 标准差: {np.std(psnr_secret):.2f}\n')
        f.write(f'   SSIM均值: {np.mean(ssim_secret):.4f}, 标准差: {np.std(ssim_secret):.4f}\n\n')
        f.write(f'=== 详细统计信息 ===\n')
        f.write(f'测试样本数量: {len(test_carriers)}\n')
        f.write(f'聚类数量: {n_clusters}\n')
        f.write(f'训练样本数量: {TRAIN_NUM}\n')
        f.write(f'测试样本数量: {TEST_NUM}\n')
    
    # 保存聚类可视化到output文件夹
    visualize_clusters(feats_for_cluster, labels, save_path=os.path.join(out_dir, 'cluster_vis.png'))
    
    # 保存一些测试样本的对比图片
    for i in range(min(5, len(test_carriers))):
        img, label = test_carriers[i], test_labels[i]
        carrier = torch.FloatTensor(img).unsqueeze(0).unsqueeze(0).to(DEVICE) / 255.0
        carrier_rgb = carrier.repeat(1,3,1,1)
        secret = torch.FloatTensor(secret_arr).unsqueeze(0).unsqueeze(0).to(DEVICE) / 255.0
        encoder = encoders[label].to(DEVICE)
        decoder = decoders[label].to(DEVICE)
        
        with torch.no_grad():
            stego = encoder(carrier_rgb, secret)
            decoded = decoder(stego)
        
        # 保存原始载体图片
        orig_img = (carrier_rgb.squeeze().cpu().numpy().transpose(1,2,0)*255).clip(0,255).astype(np.uint8)
        Image.fromarray(orig_img).save(os.path.join(out_dir, f'sample_{i}_original.png'))
        
        # 保存隐写后的图片
        stego_img = (stego.squeeze().cpu().numpy().transpose(1,2,0)*255).clip(0,255).astype(np.uint8)
        Image.fromarray(stego_img).save(os.path.join(out_dir, f'sample_{i}_stego.png'))
        
        # 保存解码后的秘密图片
        decoded_img = (decoded.squeeze().cpu().numpy()*255).clip(0,255).astype(np.uint8)
        Image.fromarray(decoded_img, mode='L').save(os.path.join(out_dir, f'sample_{i}_decoded.png'))
    
    # 保存原始秘密图片作为对比
    Image.fromarray(secret_arr).save(os.path.join(out_dir, 'original_secret.png'))
    
    print(f'所有结果已保存到 {out_dir} 文件夹')
    print('全部完成！')

if __name__ == '__main__':
    main() 