from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix

import torch


class GetConfusionMatrix:
    """
    BooDuckCamp 🦆 2021, AI Tech 2th, 3 yeared Joong-go Baby Camper
    Motivated by https://stages.ai/competitions/74/discussion/talk/post/510.
    버그를 찾으셨나요? 혹은 개선점을 찾으셨나요? 공유해주세요!
    If you find bugs, Contact @권용범_T2013 on Slack.
    """
    def __init__(self,
                 save_path: str,
                 current_epoch: int,
                 n_classes: int = 18,
                 labels: list = None,
                 tag: str = None,
                 image_name: str = 'confusion_matrix',
                 only_wrong_label: bool = True,
                 count_label: bool = False,
                 savefig: bool = True,
                 showfig: bool = False,
                 figsize: tuple = (13, 12),
                 dpi: int = 200,
                 vmax: float = None) -> None:
        """
        save_path: 저장 dir(폴더) path를 반드시 정해주세요.
        이미지가 많아질 경우 관리가 어려워질 수 있습니다.
        dir 중복은 허가하지 않습니다. 기존 이미지를 덮어 쓸 수 있습니다.
        parent_path는 허용되어 있습니다. dir이 없는 경우 생성합니다.
        """
        assert save_path, 'save_path required !!'
        assert type(current_epoch) == int, 'current_epoch required !!'

        self.save_path = save_path
        img_dir = Path(self.save_path)
        if not img_dir.exists() and savefig:
            img_dir.mkdir(parents=True)
        elif img_dir.is_file():
            raise PermissionError("지정된 image 저장 위치가 dir이 아닙니다. dir를 생성해주세요.")

        ## CLASS VAR
        self.current_epoch = current_epoch
        self.n_iter = 0
        self.n_classes = n_classes
        self.matrix = np.zeros((n_classes, n_classes))

        ## CONFUSION MATRIX
        self.only_wrong_label = only_wrong_label
        self.count_label = count_label
        if labels:
            self.labels = labels
        else:
            self.labels = [_ for _ in range(self.n_classes)]

        ## IMAGE CONFIG
        self.tag = tag
        if self.tag:
            self.tag += '.'
        self.figsize = figsize
        self.vmax = vmax
        self.dpi = dpi
        self.image_name = image_name
        self.savefig = savefig
        self.showfig = showfig

        # [BUG]
        # Document를 잘 확인하지 않으면 아래와 같은 하지 않아도될 일이 발생합니다.
        # confusion_matrix의 labels 라는 옵션을 잘 활용하면 간단히 해결됩니다.
        # """
        # **아래의 dummy confusion array가 필요합니다.**
        # 데이터가 수가 적거나 혹은 운이 좋지 않아 18개 클래스 중에서 빈 클래스가 발생할 수 있습니다.
        # 그런 경우 confusion matrix 계산시에 retrun array shape이 (18, 18)이 아닌 (17, 17)이 되는 경우가 발생합니다.
        # 기존의 순서와 다른 confusion matrix가 생길 수 있습니다. 해당 matrix를 그대로 더해주게 되면 문제됩니다.
        # 해결 법은 아래와 같습니다.
        # 아래의 dummy array를 input 되는 y_true 와 y_pred에 concatenate하여 빈 레이블이 생기는 것을 방지해 줍니다.
        # (np.ones((18, 18))와 동일한 confusion matrix 추가되어 빈 레이블이 생기지 않습니다.)
        # 그런 후 return된 array 에서 -1을 해주어 원래 계산된 값을 가지게 합니다.
        # """
        # self.n_classes = n_classes
        # dummy_class_array = np.arange(self.n_classes)
        # self.true_confusion_dummy = np.repeat(dummy_class_array,
        #                                       self.n_classes)
        # self.pred_confusion_dummy = np.tile(dummy_class_array, self.n_classes)

    def collect_batch_preds(
        self,
        y_true,
        y_pred,
    ) -> None:
        """
        기본적으로 틀린 label만 고려하여 정보를 수집합니다.
        `only_wrong_label=False`인 경우 맞고 틀린 것 모두 고려하여 confusion matrix를 계산합니다.
        `count_label=True`인 경우 normalized 된 값이 아니라 수량으로 표현됩니다.
        """
        if torch.is_tensor(y_true):
            y_true = y_true.cpu().numpy()
        if torch.is_tensor(y_pred):
            y_pred = y_pred.detach().cpu().numpy()

        if self.only_wrong_label:
            wrong_idx = (y_true != y_pred)  # 틀린 index만 확인.
            cur_cm = confusion_matrix(y_true[wrong_idx],
                                      y_pred[wrong_idx],
                                      labels=self.labels)
        else:
            cur_cm = confusion_matrix(y_true, y_pred, labels=self.labels)

        if not self.count_label:
            _div_array = np.zeros((self.n_classes, 1)) + 1e-6
            y_ture_idx, y_true_counts = np.unique(
                y_true, return_counts=True)  # Label 개수 확인.
            for _idx, _counts in zip(y_ture_idx,
                                     y_true_counts):  # To avoid empty label
                _div_array[_idx, 0] = _counts
            cur_cm = cur_cm / _div_array  # normalize

        self.matrix += cur_cm
        self.n_iter += 1

        return cur_cm

    # def _old_collect_batch_preds(
    #     self,
    #     y_true,
    #     y_pred,
    # ) -> None:
    #     """
    #     기본적으로 틀린 label만 고려하여 정보를 수집합니다.
    #     `only_wrong_label=False`인 경우 맞고 틀린 것 모두 고려하여 confusion matrix를 계산합니다.
    #     `count_label=True`인 경우 normalized 된 값이 아니라 수량으로 표현됩니다.
    #     """
    #     if torch.is_tensor(y_true):
    #         y_true = y_true.cpu().numpy()
    #     if torch.is_tensor(y_pred):
    #         y_pred = y_pred.detach().cpu().numpy()

    #     if self.only_wrong_label:
    #         wrong_idx = (y_true != y_pred)  # 틀린 index만 확인.
    #         cur_cm = confusion_matrix(
    #             np.concatenate([y_true[wrong_idx], self.true_confusion_dummy]),
    #             np.concatenate([y_pred[wrong_idx], self.pred_confusion_dummy]))
    #         cur_cm -= 1
    #     else:
    #         cur_cm = confusion_matrix(
    #             np.concatenate([y_true, self.true_confusion_dummy]),
    #             np.concatenate([y_pred, self.pred_confusion_dummy]))
    #         cur_cm -= 1

    #     if not self.count_label:
    #         _div_array = np.zeros((self.n_classes, 1)) + 1e-6
    #         y_ture_idx, y_true_counts = np.unique(
    #             y_true, return_counts=True)  # Label 개수 확인.
    #         for _idx, _counts in zip(y_ture_idx,
    #                                  y_true_counts):  # To avoid empty label
    #             _div_array[_idx, 0] = _counts
    #         cur_cm = cur_cm / _div_array  # normalize

    #     self.matrix += cur_cm
    #     self.n_iter += 1

    #     return cur_cm

    @staticmethod
    def _confusion_matrix_heatmap_plot(_confusion_matrix,
                                       _title: str = None,
                                       _figsize: tuple = None,
                                       _vmax: float = None):
        fig, ax = plt.subplots(
            1,
            1,
            figsize=_figsize,
        )
        sns.heatmap(_confusion_matrix,
                    annot=True,
                    cmap=sns.color_palette("Blues"),
                    vmax=_vmax,
                    ax=ax)
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        ax.set_title(_title)
        fig.tight_layout()
        return fig

    def epoch_plot(self, ) -> None:
        if not self.count_label:
            epoch_cm = self.matrix / self.n_iter
        else:
            epoch_cm = self.matrix

        if self.only_wrong_label:
            epoch_title = f'Confusion Matrix; only_wrong_label; epoch {self.current_epoch}'
        else:
            epoch_title = f'Confusion Matrix; epoch {self.current_epoch}'
        save_image_name = f"{self.save_path}/{self.tag}{self.image_name}.epoch{self.current_epoch}.png"
        fig = self._confusion_matrix_heatmap_plot(epoch_cm,
                                                  _title=epoch_title,
                                                  _figsize=self.figsize,
                                                  _vmax=self.vmax)
        if self.savefig == "png":
            fig.savefig(save_image_name, dpi=self.dpi)
            plt.close(fig)
        # modified
        elif self.savefig == "tensorboard":
            import io
            buf = io.BytesIO()
            fig.savefig(buf, format = 'jpeg')
            buf.seek(0)
            self.plot_buf = buf
            plt.close(fig)
        #####
        
        if self.showfig:
            fig.show()

    def get_stat(self) -> np.array:
        if not self.count_label:
            cm = self.matrix / self.n_iter
        else:
            cm = self.matrix  # dummy function
        label_error = cm.sum(axis=-1)
        return label_error


if __name__ == "__main__":

    from get_confusion_matrix import GetConfusionMatrix

    NUM_CLASS = 18

    for epoch in range(1):
        label_cm = GetConfusionMatrix(
            save_path='confusion_matrix_image',
            current_epoch=epoch,  # 구분점을 epoch으로 두었습니다. (반드시 Epoch일 필요 X)
            n_classes=NUM_CLASS,
            tag='wrong_False_count_False',  # for multi-model
            # image_name='confusion_matrix',  # default file name
            # only_wrong_label=False,  # wrong label만 표현합니다. (default: True)
            # count_label=True,  # 수량으로 표현합니다.(default: False)
            # savefig=False,  # for jupyter-notebook (default: True)
            # showfig=True,  # for jupyter-notebook (default: False)
            figsize=(13, 12),  # <- default figsize
            # dpi=200,  # Matplotlib's default is 150 dpi. (default: 200)
            vmax=None)  # A max value of colorbar of heatmap

        for _ in range(5):  # dummy Dataloader
            # train
            target = torch.randint(0, NUM_CLASS, (32, ))
            pred = torch.randint(0, NUM_CLASS, (32, ))
            # prediction
            label_cm.collect_batch_preds(target, pred)

        label_cm.epoch_plot()
