import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
from trans import JSSPTransformer
import os

EXPERIMENT_NAME = 'jssp_transformer_0'

# 체크포인트 저장 함수
def save_checkpoint(state, filename):
    torch.save(state, filename)

# 학습 함수
def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    train_loss = 0
    for batch_idx, (x_batch, y_batch) in enumerate(train_loader):
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        output = model(x_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        
        if batch_idx % 10 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(x_batch)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
    
    avg_train_loss = train_loss / len(train_loader)
    writer.add_scalar('Loss/train', avg_train_loss, epoch)
    return avg_train_loss

# 검증 함수
def validate(model, device, val_loader, criterion, epoch):
    model.eval()
    val_loss = 0
    correct = 0
    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            output = model(x_batch)
            val_loss += criterion(output, y_batch).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(y_batch.view_as(pred)).sum().item()
    
    avg_val_loss = val_loss / len(val_loader)
    accuracy = correct / len(val_loader.dataset)
    writer.add_scalar('Loss/validation', avg_val_loss, epoch)
    writer.add_scalar('Accuracy/validation', accuracy, epoch)
    return avg_val_loss, accuracy

if __name__ == '__main__':
    # 데이터 로드
    data = np.load('/media/NAS/USERS/moonbo/jssp/jssp_supervision_data.npz')
    x = data['x']
    y = data['y']

    # 데이터를 torch tensor로 변환
    x_tensor = torch.tensor(x, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)

    # 데이터셋 생성
    dataset = TensorDataset(x_tensor, y_tensor)

    # 데이터셋을 8:1:1 비율로 학습, 검증, 테스트 세트로 분할
    train_x, test_x, train_y, test_y = train_test_split(x_tensor, y_tensor, test_size=0.2, random_state=42)
    val_x, test_x, val_y, test_y = train_test_split(test_x, test_y, test_size=0.5, random_state=42)

    # TensorDataset으로 변경
    train_dataset = TensorDataset(train_x, train_y)
    val_dataset = TensorDataset(val_x, val_y)
    test_dataset = TensorDataset(test_x, test_y)

    # 데이터 로더 생성
    batch_size = 32

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Parameters
    job_count = 100          # Number of unique jobs
    machine_count = 20       # Number of unique machines = max operations
    d_model = 64             # Embedding dimension
    num_heads = 8            # Number of attention heads
    num_layers = 6           # Number of transformer encoder layers
    num_heuristics = 8       # Number of heuristics to predict

    # 디바이스 설정
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)

    # 모델 초기화
    model = JSSPTransformer(job_count, machine_count, d_model, num_heads, num_layers, num_heuristics)
    model = model.to(device)

    # 손실 함수 및 옵티마이저
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # TensorBoard SummaryWriter 생성
    writer = SummaryWriter()

    checkpoint_folder = os.path.join('/media/NAS/USERS/moonbo/jssp/jssp/checkpoints/', EXPERIMENT_NAME)
    os.makedirs(checkpoint_folder, exist_ok=True)

    # 학습 및 검증 루프
    num_epochs = 100
    best_val_loss = float('inf')
    for epoch in range(1, num_epochs + 1):
        train_loss = train(model, device, train_loader, optimizer, criterion, epoch)
        val_loss, val_accuracy = validate(model, device, val_loader, criterion, epoch)
        print(f'Epoch {epoch}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')
        
        # 체크포인트 저장
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            save_checkpoint({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'best_val_loss': best_val_loss,
                'optimizer': optimizer.state_dict(),
            }, filename=os.path.join(checkpoint_folder, 'best_checkpoint.pth'))
        
        if epoch % 5 == 0:
            save_checkpoint({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'best_val_loss': best_val_loss,
                'optimizer': optimizer.state_dict(),
            }, filename=os.path.join(checkpoint_folder, f'checkpoint_epoch_{epoch}.pth'))

    # 테스트
    test_loss, test_accuracy = validate(model, device, test_loader, criterion, num_epochs)
    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')

    # TensorBoard SummaryWriter 종료
    writer.close()
