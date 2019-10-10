import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Embedder(nn.Module):
    ''' id -> 単語分散表現ベクトル '''
    def __init__(self, text_embedding_vectors):
        super(Embedder, self).__init__()
        self.embeddings = nn.Embedding.from_pretrained(
            embeddings=text_embedding_vectors,
            freeze=True # バックプロパゲーションで更新されなくなる
        )
    
    def forward(self, x):
        x_vec = self.embeddings(x)
        return x_vec


class PositionalEncoder(nn.Module):
    ''' 入力された単語の位置を示すベクトル情報を付加 '''
    def __init__(self, device, d_model=300, max_seq_len=256):
        super().__init__()

        self.d_model = d_model
        pe = torch.zeros(max_seq_len, d_model)
        
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i) / d_model) ))
                pe[pos, i + 1] = math.cos(pos / 10000 ** ((2 * (i+1)) / d_model))
        
        # peの先頭にミニバッチの次元を足す
        self.pe = pe.unsqueeze(0)
        
        # 位置情報テンソルは勾配を計算しない
        self.pe.requires_grad = False
        
        # peをdeviceに渡す
        self.pe = pe.to(device)
    
    def forward(self, x):
        ret = math.sqrt(self.d_model) * x + self.pe
        return ret


class Attention(nn.Module):
    def __init__(self, d_model=300):
        super().__init__()
        
        # SAGANでは1dConvを使用したが，ここでは全結合層を使う
        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        
        # 出力時に使用する全結合層
        self.out = nn.Linear(d_model, d_model)
        
        # Attentionの大きさ調整の変数
        self.d_k = d_model
    
    def forward(self, q, k, v, mask):
        # 全結合層で特徴量を変換
        k = self.k_linear(k)
        q = self.q_linear(q)
        v = self.v_linear(v)
        
        # Attentionの値を計算
        # 各値を足し算すると大きくなりすぎるのでroot(d_k)で割って調整
        weights = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(self.d_k)
        
        # ここでmaskを計算
        # 0になっているところを大きな負の値にし，Attentionがかからないようにする
        mask = mask.unsqueeze(1)
        weights = weights.masked_fill(mask == 0, -1e9)
        
        # softmaxで行ごとの規格化を行う
        normalized_weights = F.softmax(weights, dim=-1)
        
        # AttentionをValueと掛け算
        output = torch.matmul(normalized_weights, v)
        
        # 全結合層で特徴量を今一度変換
        output = self.out(output)
        return output, normalized_weights


class FeedForward(nn.Module):
    ''' Attention層からの出力を単純に全結合層2つで特徴量変換するユニット '''
    def __init__(self, d_model, d_ff=1024, dropout=0.1):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)
    
    def forward(self, x):
        x = self.linear_1(x)
        x = self.dropout(F.relu(x))
        x = self.linear_2(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        
        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)
        
        self.attn = Attention(d_model)
        
        self.ff = FeedForward(d_model)
        
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
    
    def forward(self, x, mask):
        # 正規化とAttention
        x_normalized = self.norm_1(x)
        output, normalized_weights = self.attn(
            x_normalized, x_normalized, x_normalized, mask
        )
        
        x2 = x + self.dropout_1(output)
        
        # 正規化と全結合層
        x_normalized2 = self.norm_2(x2)
        output = x2 + self.dropout_2(self.ff(x_normalized))
        
        return output, normalized_weights


class ClassificationHead(nn.Module):
    def __init__(self, d_model=300, output_dim=2):
        super().__init__()
        
        # 全結合層
        self.linear = nn.Linear(d_model, output_dim)
        
        # 重みを平均0,標準偏差0.02の正規分布で, バイアスを0で初期化する
        nn.init.normal_(self.linear.weight, std=0.02)
        nn.init.normal_(self.linear.bias, 0)
        
    def forward(self, x):
        # 各ミニバッチの先頭の単語の特徴量(300次元)のみを取り出す
        x0 = x[:, 0, :]
        out = self.linear(x0)
        return out


class TransformerClassification(nn.Module):
    ''' Transformerでクラス分類させる '''
    def __init__(self, device, text_embedding_vectors, 
                 d_model=300, max_seq_len=256, output_dim=2):
        super().__init__()
        
        # モデル構築
        self.net1 = Embedder(text_embedding_vectors)
        self.net2 = PositionalEncoder(device, d_model=d_model, max_seq_len=max_seq_len)
        self.net3_1 = TransformerBlock(d_model=d_model)
        self.net3_2 = TransformerBlock(d_model=d_model)
        self.net4 = ClassificationHead(output_dim=output_dim, d_model=d_model)
    
    def forward(self, x, mask):
        x1 = self.net1(x)
        x2 = self.net2(x1)
        x3_1, normalized_weights_1 = self.net3_1(x2, mask)
        x3_2, normalized_weights_2 = self.net3_2(x3_1, mask)
        x4 = self.net4(x3_2)
        
        return x4, normalized_weights_1, normalized_weights_2