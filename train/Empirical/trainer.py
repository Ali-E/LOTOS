import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
import time
import torch.nn.functional as F
import torch.autograd as autograd
import sys
import os
import numpy as np

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(os.path.dirname(currentdir))
sys.path.append(parentdir)


from utils.Empirical.utils_ensemble import AverageMeter, requires_grad_
from models.ensemble import Ensemble
from utils.Empirical.utils_ensemble import Cosine, Magnitude


def PGD(models, inputs, labels, eps):
    steps = 6
    alpha = eps / 3.

    adv = inputs.detach() + torch.FloatTensor(inputs.shape).uniform_(-eps, eps).cuda()
    adv = torch.clamp(adv, 0, 1)
    criterion = nn.CrossEntropyLoss()

    adv.requires_grad = True
    for _ in range(steps):
        grad_loss = 0
        for i, m in enumerate(models):
            loss = criterion(m(adv), labels)
            grad = autograd.grad(loss, adv, create_graph=True)[0]
            grad = grad.flatten(start_dim=1)
            grad_loss += Magnitude(grad)

        grad_loss /= 3
        grad_loss.backward()
        sign_grad = adv.grad.data.sign()
        with torch.no_grad():
            adv.data = adv.data + alpha * sign_grad
            adv.data = torch.max(torch.min(adv.data, inputs + eps), inputs - eps)
            adv.data = torch.clamp(adv.data, 0., 1.)

    adv.grad = None
    return adv.detach()


def Naive_Trainer_ortho_catclip_2(args, loader: DataLoader, models, criterion, optimizer: Optimizer, epoch: int, device: torch.device, writer=None, catclip=False, no_effect_epochs=0, batch_counter=0, mal_freq=100, conv_freq=50, layer_1_only=False, conv_1st_only=False, lsv_list_dict={}, lsv_list_dict_conv={}, conv_only=False, cos_loss_flag=False, trs_flag=False, gal_flag=False):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    reg_losses = AverageMeter()
    ortho_losses = AverageMeter()

    if gal_flag:
        gal_losses = AverageMeter()

    if cos_loss_flag or trs_flag:
        cos_losses = AverageMeter()
        cos01_losses = AverageMeter()
        cos02_losses = AverageMeter()
        cos12_losses = AverageMeter()

    if trs_flag:
        smooth_losses = AverageMeter()

    end = time.time()
    ortho_total = 0.

    ortho_flag = True
    if args.conv_factor < 0.000001 and args.cat_factor < 0.000001:
        ortho_flag = False

    decrement = args.decrement
    weights = torch.from_numpy(np.array([1 - decrement*i for i in range(100)])).to(device)

    print_freq = max(1000, 10*conv_freq)
    if len(lsv_list_dict) == 0:
        print('initiating lsv list dict')
        for j in range(args.num_models):
            lsv_list_dict[j] = None
            lsv_list_dict_conv[j] = None

    for i in range(args.num_models):
        models[i].train()
        requires_grad_(models[i], True)

    cat_counter_info = 0
    conv_counter_info = 0
    for i, (inputs, targets) in enumerate(loader):
        data_time.update(time.time() - end)
        inputs, targets = inputs.to(device), targets.to(device)
        batch_size = inputs.size(0)
        loss_std = 0
        ortho_loss = 0
        ortho_loss_conv = 0

        grads = []
        if cos_loss_flag or gal_flag or trs_flag:
            inputs.requires_grad = True

        for j in range(args.num_models):
            logits = models[j](inputs)
            loss = criterion(logits, targets)
            
            if cos_loss_flag or gal_flag or trs_flag:
                grad = autograd.grad(loss, inputs, create_graph=True)[0]
                grad = grad.flatten(start_dim=1)
                grads.append(grad)

            loss_std += loss

            if not ortho_flag:
                continue

            if i == len(loader)-1:
                continue

            VT_list = []
            VT_list_conv = []
            idx = 0

            conv_count = 0
            cat_counter_info = 0
            conv_counter_info = 0

            for (m_name, m) in models[j].named_modules():
                condition = isinstance(m, (torch.nn.Conv2d))
                condition_conv = isinstance(m, (torch.nn.Conv2d))
                if not condition_conv and conv_only:
                    continue
                if catclip:
                    condition = not conv_only and not isinstance(m, (torch.nn.Conv2d)) and (not isinstance(m, torch.nn.BatchNorm2d) and not isinstance(m, torch.nn.Linear))

                if not condition_conv and epoch < no_effect_epochs:
                    conv_factor = 0.0
                    cat_factor = 0.0
                else:
                    conv_factor = args.conv_factor
                    cat_factor = args.cat_factor

                if condition or condition_conv:
                    attrs = vars(m)
                    for item in attrs.items():
                        if item[0] == '_buffers' and 'weight_VT' in item[1]:
                            if epoch == 0 and i == 0:
                                if condition_conv:
                                    print('model: ', j, 'conv_count: ', idx)
                                else:
                                    print('model: ', j, 'idx: ', idx)
                            VT = item[1]['weight_VT']
                            if batch_counter != 0:# or epoch == no_effect_epochs:# and j != 0:
                                for k in range(args.num_models):
                                    if k == j:
                                        if condition_conv:
                                            if batch_counter % print_freq != 0:
                                                continue
                                            prev_VT = lsv_list_dict_conv[k]
                                            sing_vector = prev_VT[conv_count]
                                        else:
                                            if batch_counter % mal_freq != 0:
                                                continue
                                            prev_VT = lsv_list_dict[k]
                                            sing_vector = prev_VT[idx]

                                        sing_vector = torch.nn.parameter.Parameter(data=sing_vector, requires_grad=False)
                                        op_shape = [i for i in range(1, len(sing_vector.shape))]
                                        continue

                                    if condition_conv:
                                        prev_VT_list = lsv_list_dict_conv[k]
                                    else:
                                        prev_VT_list = lsv_list_dict[k]

                                    if not condition_conv and layer_1_only and idx > 0:
                                        continue

                                    if condition_conv and conv_1st_only and conv_count > 0:
                                        continue

                                    if condition_conv:
                                        bad_vector = prev_VT_list[conv_count]
                                    else:
                                        bad_vector = prev_VT_list[idx]

                                    if batch_counter % mal_freq != 0 and not condition_conv:
                                        continue
                                    
                                    if batch_counter % conv_freq != 0 and condition_conv:
                                        continue

                                    bad_vector = torch.nn.parameter.Parameter(data=bad_vector, requires_grad=False)
                                    op_shape = [i for i in range(1, len(bad_vector.shape))]
                                    bad_vec_length = torch.sqrt(torch.sum((m(bad_vector) - m(torch.zeros_like(bad_vector)) )**2, axis=op_shape))/torch.sqrt(torch.sum(bad_vector **2, axis=op_shape)) ##### fix this shit for multiple vectors!                       

                                    if condition_conv:
                                        bad_vec_length_thresh = torch.nn.functional.relu(bad_vec_length-args.bottom_clip)
                                    else:
                                        bad_vec_length_thresh = torch.nn.functional.relu(bad_vec_length-args.cat_bottom_clip)

                                    bad_vec_length_weighted = torch.sum(torch.mul(bad_vec_length_thresh, weights[:len(bad_vec_length_thresh)]))/torch.sum(weights[:len(bad_vec_length_thresh)])

                                    if condition_conv:
                                        ortho_loss_conv += conv_factor * bad_vec_length_weighted
                                        conv_counter_info += 1
                                    if not condition_conv:
                                        ortho_loss += cat_factor * bad_vec_length_weighted
                                        cat_counter_info += 1

                            if condition_conv:
                                VT_list_conv.append(VT.detach())
                                conv_count += 1
                            else:
                                VT_list.append(VT.detach())
                                idx += 1

            lsv_list_dict_conv[j] = VT_list_conv
            if not conv_only:
                lsv_list_dict[j] = VT_list

        reg_losses.update(loss_std.item(), batch_size)
        loss = loss_std

        if gal_flag:
            cos_sim = []
            for ii in range(len(models)):
                for j in range(ii + 1, len(models)):
                        cos_sim.append(F.cosine_similarity(grads[ii], grads[j], dim=-1))

            cos_sim = torch.stack(cos_sim, dim=-1)
            gal_loss = torch.log(cos_sim.exp().sum(dim=-1) + 1e-20).mean()
            loss += 0.5 * gal_loss

        cos_loss, smooth_loss = 0, 0

        if cos_loss_flag or trs_flag:
            cos01 = Cosine(grads[0], grads[1])
            cos02 = Cosine(grads[0], grads[2])
            cos12 = Cosine(grads[1], grads[2])
            cos_loss = (cos01 + cos02 + cos12) / 3.

            if trs_flag:
                loss += args.scale * args.coeff * cos_loss
            else:
                loss += 0.5 * cos_loss
        
        if trs_flag and (batch_counter % conv_freq) != 0:
            if (args.plus_adv):
                if i == 0:
                    print('full TRS method!')

                N = inputs.shape[0] // 2
                cureps = (args.trs_adv_eps - args.init_eps) * epoch / args.epochs + args.init_eps
                clean_inputs = inputs[:N].detach()	# PGD(self.models, inputs[:N], targets[:N])
                adv_inputs = PGD(models, inputs[N:], targets[N:], cureps).detach()
                adv_x = torch.cat([clean_inputs, adv_inputs])

                adv_x.requires_grad = True
                for j in range(args.num_models):
                    outputs = models[j](adv_x)
                    loss = criterion(outputs, targets)
                    grad = autograd.grad(loss, adv_x, create_graph=True)[0]
                    grad = grad.flatten(start_dim=1)
                    smooth_loss += Magnitude(grad)

            else:
                if i == 0:
                    print('TRS l2 method!')
                for j in range(args.num_models):
                    outputs = models[j](inputs)
                    loss = criterion(outputs, targets)
                    grad = autograd.grad(loss, inputs, create_graph=True)[0]
                    grad = grad.flatten(start_dim=1)
                    smooth_loss += Magnitude(grad)

            smooth_loss /= 3
            loss += args.scale * args.lamda * smooth_loss

        if ortho_flag:
            pair_count = args.num_models * (args.num_models - 1) / 2
            conv_counter_info = conv_counter_info // (args.num_models-1)
            cat_counter_info = cat_counter_info // (args.num_models-1)
            
            conv_normalizer = conv_counter_info*pair_count
            cat_normalizer = cat_counter_info*pair_count

            conv_normalizer = 1.
            cat_normalizer = 1.

            if ortho_loss_conv > 0 and (batch_counter % 200 != 199) and batch_counter > 0 and conv_counter_info > 0 and (batch_counter % conv_freq) == 0:
                loss += ortho_loss_conv / conv_normalizer #/ (conv_counter_info*pair_count)
                ortho_total += ortho_loss_conv.item() / conv_normalizer #/ (conv_counter_info*pair_count)
                if batch_counter % print_freq == 0:
                    print('pairs', pair_count,  'conv', conv_counter_info,  'ortho loss conv: ', ortho_loss_conv.item()/ conv_normalizer )

            if not conv_only and batch_counter > 0 and cat_counter_info > 0 and (batch_counter % mal_freq) == 50:
                loss += ortho_loss / cat_normalizer #/ (cat_counter_info*pair_count)
                ortho_total += ortho_loss.item() / cat_normalizer #/ (cat_counter_info*pair_count)
                if batch_counter % mal_freq == 0:
                    print('pairs', pair_count, 'cat', cat_counter_info, 'ortho loss cat: ', ortho_loss.item() / cat_normalizer)
        
        losses.update(loss.item(), batch_size)

        if cos_loss_flag or trs_flag:
            cos01_losses.update(cos01.item(), batch_size)
            cos02_losses.update(cos02.item(), batch_size)
            cos12_losses.update(cos12.item(), batch_size)
            cos_losses.update(cos_loss.item(), batch_size)

        if trs_flag and smooth_loss > 0:
            smooth_losses.update(smooth_loss.item(), batch_size)

        if gal_flag:
            gal_losses.update(gal_loss.item(), batch_size)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
    
        batch_counter += 1

    print('Epoch: ', epoch, 'Loss: ', losses.avg, 'Loss_std: ', reg_losses.avg, 'Ortho_loss: ', ortho_losses.avg, 'ortho total:', ortho_total)

    writer.add_scalar('train/batch_time', batch_time.avg, epoch)
    writer.add_scalar('train/loss', losses.avg, epoch)
    writer.add_scalar('train/reg_loss', reg_losses.avg, epoch)
    writer.add_scalar('train/ortho_loss', ortho_losses.avg, epoch)

    return losses.avg, batch_counter, lsv_list_dict, lsv_list_dict_conv
