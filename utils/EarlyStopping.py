class EarlyStopping:
    def __init__(self, patience=50, min_delta=0, start_epoch=50):
        self.patience = patience        # 允许loss不改善的epoch数
        self.min_delta = min_delta      # 视为有效提升的最小变化值
        self.start_epoch = start_epoch  # 不会早停的最小epoch数。start_epoch个epoch以后才会执行早停机制，避免训练早期验证集抖动导致提前停止。
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False

    def __call__(self, eval_loss, epoch):
        if (self.best_loss - eval_loss) > self.min_delta:
            # 最优模型
            self.best_loss = eval_loss
            self.counter = 0
        else:
            # 不是最优模型
            self.counter += 1
            if self.counter >= self.patience and (epoch > self.start_epoch):
                self.early_stop = True
        return self.early_stop
