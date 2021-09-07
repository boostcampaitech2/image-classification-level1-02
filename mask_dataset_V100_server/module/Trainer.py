import os
import re
import PIL
import torch
import numpy as np
from torchvision import transforms
from module.get_confusion_matrix import GetConfusionMatrix


def train_loop(
        model,
        hp,
        device,
        dataset,
        train_dataloader,
        val_dataloader,
        loss_combination,
        opt,
        scheduler,
        tr_writer,
        val_writer,
        hist_writer
):
    global_step = 0
    for ep in range(hp.TOTAL_EPOCH + 1):

        # = Training phase =========
        tr_mean_loss, tr_mean_f1, tr_acc = 0, 0, 0
        for X, y in iter(train_dataloader):
            global_step += 1

            # = Zero epoch recording ==================
            if not ep:
                with torch.no_grad():
                    model.eval()
                    predict = model(X.to(device))

                    loss_val = loss_combination(predict, y.to(device))
                    tr_mean_loss += loss_val
                    tr_mean_f1 += (1 - loss_combination.loss_2_val)

                    _, argmax = torch.max(predict.data, 1)
                    tr_acc += (argmax == y.to(device)).sum().item() / hp.BATCH_SIZE

            # = train epoch recording ==================
            else:
                model.train()
                predict = model(X.to(device))

                loss_val = loss_combination(predict, y.to(device))
                tr_mean_loss += loss_val
                tr_mean_f1 += (1 - loss_combination.loss_2_val)

                _, argmax = torch.max(predict.data, 1)
                tr_acc += (argmax == y.to(device)).sum().item() / hp.BATCH_SIZE

                # update
                opt.zero_grad()
                loss_val.backward()
                opt.step()
        tr_mean_loss = tr_mean_loss / len(train_dataloader)
        tr_mean_f1 = tr_mean_f1 / len(train_dataloader)
        tr_acc = tr_acc / len(train_dataloader)

        # = Training writer =========
        if bool(tr_writer):
            tr_writer.add_scalar('loss/CE', tr_mean_loss, ep)
            tr_writer.add_scalar('score/acc', tr_acc, ep)
            tr_writer.add_scalar('score/F1', tr_mean_f1, ep)

        if hp.SCHEDULER:
            scheduler.step()

        if hp.SPLIT:
            # = Validation phase =============

            label_cm = GetConfusionMatrix(
                save_path='confusion_matrix_image',
                current_epoch=ep,
                n_classes=len(dataset.classes),
                only_wrong_label=False,
                savefig="tensorboard",
                tag=hp.EXP_NUM,
                image_name='confusion_matrix',
            )

            val_mean_loss, val_mean_f1, val_acc = 0, 0, 0
            with torch.no_grad():
                for X, y in iter(val_dataloader):
                    model.eval()
                    predict = model(X.to(device))

                    loss_val = loss_combination(predict, y.to(device))
                    val_mean_loss += loss_val
                    val_mean_f1 += (1 - loss_combination.loss_2_val)

                    _, argmax = torch.max(predict.data, 1)
                    val_acc += (argmax == y.to(device)).sum().item() / hp.BATCH_SIZE

                    label_cm.collect_batch_preds(
                        y.to(device),
                        torch.max(predict, dim=1)[1]
                    )

            label_cm.epoch_plot()
            image = PIL.Image.open(label_cm.plot_buf)
            image = transforms.ToTensor()(image).unsqueeze(0)

            val_mean_loss = val_mean_loss / len(val_dataloader)
            val_mean_f1 = val_mean_f1 / len(val_dataloader)
            val_acc = val_acc / len(val_dataloader)
            
            # = Validation writer =========
            if bool(val_writer):
                val_writer.add_scalar('loss/CE', val_mean_loss, ep)
                val_writer.add_scalar('score/acc', val_acc, ep)
                val_writer.add_scalar('score/F1', val_mean_f1, ep)
                val_writer.add_images('CM/comfusion_matrix', image, global_step=ep)
            
        # = histogram =================
        if hp.HIST_LOG:
            for param_name, param in model.named_parameters():
                if param.requires_grad:
                    tag = ""
                    if re.search("weight", param_name):
                        tag += 'weight/'
                    elif re.search("bias", param_name):
                        tag += 'bias/'
                    hist_writer.add_histogram(
                        tag=tag + param_name,
                        values=param,
                        global_step=ep,
                    )

        print("ep : ", ep, end='\r')
        
        if hp.SAVE_MODEL:
            saved_model_path = './saved_model/model_%s/model/' % hp.EXP_NUM
            os.makedirs(saved_model_path, exist_ok=True)
            torch.save(model, saved_model_path + 'ep_%d.pt' % ep)
        
        if hp.SAVE_WEIGHT:
            saved_weights_path = './saved_model/model_%s/weights/' % hp.EXP_NUM
            os.makedirs(saved_weights_path, exist_ok=True)
            torch.save(model.state_dict(), saved_weights_path + '/ep_%d.pt' % ep)

    if hp.SPLIT:
        val_writer.add_hparams(
            hp.__dict__,
            {
                "loss/CE"   : None,
                "score/F1"  : None,
                "score/acc" : None,
            },
            run_name=None
        )
