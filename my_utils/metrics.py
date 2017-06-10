import numpy as np



def precision_recall_curve_threhold(iou_list,gt_masks,confidence):
    threhold_list = np.arange(0,3,0.05)
    precision_recall_per_img = []
    TP_FN = float(gt_masks.shape[2])
    a = zip(confidence, range(len(confidence)))
    a.sort(key=lambda x: x[0])
    index = [x[1] for x in a[::-1]]

    for i_th in range(len(threhold_list)):
        proposal_num = 0
        flag = [0] * int(TP_FN)
        cls = [-1] * len(index)
        ### calculate proposal_num satisfied
        for i in index:
            if confidence[i] > threhold_list[i_th]:
                proposal_num += 1
            else:
                break

        ### caculate per threhold
        # object
        if proposal_num != 0:
            for i_ob in range(int(TP_FN)):
                max = 0
                max_id = -1
                # proposal
                for j in index[:proposal_num]:
                    if cls[j] != 1:
                        cls[j] = 0
                    if iou_list[j][i_ob] > 0.5 and iou_list[j][i_ob] > max:
                        max = iou_list[j][i_ob]
                        max_id = j
                if max_id != -1:
                    cls[max_id] = 1
                    flag[i_ob] = 1
            TP = float(cls.count(1))
            FP = float(cls.count(0))
            TP_r = float(flag.count(1))
            precison = TP / (TP + FP)
            recall = TP_r / TP_FN
            precision_recall_per_img.append((precison, recall))
        else:
            precision_recall_per_img.append((-1, -1))

    return precision_recall_per_img

def precision_recall_curve_threhold_new(iou_list,gt_masks,confidence):
    TP_FP_FN_list = []
    threhold_list = np.arange(0,3,0.05)
    precision_recall_per_img = []
    TP_FN = float(gt_masks.shape[2])
    a = zip(confidence, range(len(confidence)))
    a.sort(key=lambda x: x[0])
    index = [x[1] for x in a[::-1]]

    for i_th in range(len(threhold_list)):
        proposal_num = 0
        flag = [0] * int(TP_FN)
        cls = [-1] * len(index)
        ### calculate proposal_num satisfied
        for i in index:
            if confidence[i] > threhold_list[i_th]:
                proposal_num += 1
            else:
                break

        ### caculate per threhold
        # object
        if proposal_num != 0:
            for i_ob in range(int(TP_FN)):
                # proposal
                for j in index[:proposal_num]:
                    if cls[j]!=1:
                        cls[j] = 0
                    if iou_list[j][i_ob] > 0.5:
                        cls[j] = 1
                        flag[i_ob] = 1
            TP = float(cls.count(1))
            FP = float(cls.count(0))
            FN = float(flag.count(0))
            precison = TP / (TP + FP)
            recall = TP / (TP + FN)
            precision_recall_per_img.append((precison, recall))
            TP_FP_FN_list.append((TP,FP,FN))
        else:
            precision_recall_per_img.append((-1, -1))
            TP_FP_FN_list.append((0,0,TP_FN))

    return precision_recall_per_img


def precision_recall_curve_threhold_new_all(iou_list,gt_masks,confidence):
    TP_FP_FN_list = []
    threhold_list = np.arange(0,5,0.05)
    precision_recall_per_img = []
    TP_FN = float(gt_masks.shape[2])
    a = zip(confidence, range(len(confidence)))
    a.sort(key=lambda x: x[0])
    index = [x[1] for x in a[::-1]]

    for i_th in range(len(threhold_list)):
        proposal_num = 0
        flag = [0] * int(TP_FN)
        cls = [-1] * len(index)
        ### calculate proposal_num satisfied
        for i in index:
            if confidence[i] > threhold_list[i_th]:
                proposal_num += 1
            else:
                break

        ### caculate per threhold
        # object
        if proposal_num != 0:
            for i_ob in range(int(TP_FN)):
                # proposal
                for j in index[:proposal_num]:
                    if cls[j]!=1:
                        cls[j] = 0
                    if iou_list[j][i_ob] > 0.5:
                        cls[j] = 1
                        flag[i_ob] = 1
            TP = float(cls.count(1))
            FP = float(cls.count(0))
            FN = float(flag.count(0))
            precison = TP / (TP + FP)
            recall = TP / (TP + FN)
            precision_recall_per_img.append((precison, recall))
            TP_FP_FN_list.append((TP,FP,FN))
        else:
            precision_recall_per_img.append((-1, -1))
            TP_FP_FN_list.append((0,0,TP_FN))

    return precision_recall_per_img, TP_FP_FN_list



def voc_ap(rec, prec, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap
