# Self-Attention ConvLSTM for Spatiotemporal Prediction

## Author

Zhihui Lin,1,2 Maomao Li,2 Zhuobin Zheng,1,2 Yangyang Cheng,1,2 Chun Yuan2,3*
1Department of Computer Science and Technologies, Tsinghua University, Beijing,
China 2Graduate School at Shenzhen, Tsinghua University, Shenzhen, China 3Peng
Cheng Laboratory, Shenzhen, China {lin-zh14, mm-li17, zhengzb16,
cheng-yy13}@mails.tsinghua.edu.cn, yuanc@sz.tsinghua.edu.cn

## Journal

Arxiv

## Motivation

---

Standard ConvLSTM cells memorize the spatial appearances and ConvLSTM always
relys on the convolution layers to capture the spatial dependence, which are
local and inefficient.

To extract spatial features with both gloal and local, they introduce the
self-attention mechanism into ConvLSTM.

## Method

---

They proposed a novel self-attention memory named as SAM. SAM is proposed to
memorize features with long-range dependences in terms of spatial and temporal
domains.

SAM are based on self-attention and also can produce features by aggregating
features across all positions of both the input self and memory features with
pair-wise similarity scores. The additional memory is updated by a gating
mechanism on aggregated features and an established highway with the memory of
th previous time step.

Technically, they prepare Memory (M) to obtain lon-range dependances more
efficiently for self-attention. And update current hiddnen layer and past memory
using self attention to extract global spatial features.

## Insight

---

The proposed model SAM-ConvLSTM had higher accuracy than normal ConvLSTM or
PredRNN and any other similar models for the three datasets (MvingMNIST, TexiBJ
and KTH).

## Contribution Summary

---

## Keyword

---

## Unknown

---

## Reflection

---

## Reference

---
