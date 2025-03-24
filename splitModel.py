GraphModule(
    (transformer): Module(
    (word_embeddings): Embedding(250880, 1024)
(word_embeddings_layernorm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
(h): Module(
    (0): Module(
    (input_layernorm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
(self_attention): Module(
    (query_key_value): Linear(in_features=1024, out_features=3072, bias=True)
(attention_dropout): Dropout(p=0.0, inplace=False)
(dense): Linear(in_features=1024, out_features=1024, bias=True)
)
(post_attention_layernorm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
(mlp): Module(
    (dense_h_to_4h): Linear(in_features=1024, out_features=4096, bias=True)
(dense_4h_to_h): Linear(in_features=4096, out_features=1024, bias=True)
)
)
(1): Module(
    (input_layernorm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
(self_attention): Module(
    (query_key_value): Linear(in_features=1024, out_features=3072, bias=True)
(attention_dropout): Dropout(p=0.0, inplace=False)
(dense): Linear(in_features=1024, out_features=1024, bias=True)
)
(post_attention_layernorm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
(mlp): Module(
    (dense_h_to_4h): Linear(in_features=1024, out_features=4096, bias=True)
(dense_4h_to_h): Linear(in_features=4096, out_features=1024, bias=True)
)
)
(2): Module(
    (input_layernorm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
(self_attention): Module(
    (query_key_value): Linear(in_features=1024, out_features=3072, bias=True)
(attention_dropout): Dropout(p=0.0, inplace=False)
(dense): Linear(in_features=1024, out_features=1024, bias=True)
)
(post_attention_layernorm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
(mlp): Module(
    (dense_h_to_4h): Linear(in_features=1024, out_features=4096, bias=True)
(dense_4h_to_h): Linear(in_features=4096, out_features=1024, bias=True)
)
)
(3): Module(
    (input_layernorm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
(self_attention): Module(
    (query_key_value): Linear(in_features=1024, out_features=3072, bias=True)
(attention_dropout): Dropout(p=0.0, inplace=False)
(dense): Linear(in_features=1024, out_features=1024, bias=True)
)
(post_attention_layernorm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
(mlp): Module(
    (dense_h_to_4h): Linear(in_features=1024, out_features=4096, bias=True)
(dense_4h_to_h): Linear(in_features=4096, out_features=1024, bias=True)
)
)
(4): Module(
    (input_layernorm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
(self_attention): Module(
    (query_key_value): Linear(in_features=1024, out_features=3072, bias=True)
(attention_dropout): Dropout(p=0.0, inplace=False)
(dense): Linear(in_features=1024, out_features=1024, bias=True)
)
(post_attention_layernorm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
(mlp): Module(
    (dense_h_to_4h): Linear(in_features=1024, out_features=4096, bias=True)
(dense_4h_to_h): Linear(in_features=4096, out_features=1024, bias=True)
)
)
(5): Module(
    (input_layernorm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
(self_attention): Module(
    (query_key_value): Linear(in_features=1024, out_features=3072, bias=True)
(attention_dropout): Dropout(p=0.0, inplace=False)
(dense): Linear(in_features=1024, out_features=1024, bias=True)
)
(post_attention_layernorm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
(mlp): Module(
    (dense_h_to_4h): Linear(in_features=1024, out_features=4096, bias=True)
(dense_4h_to_h): Linear(in_features=4096, out_features=1024, bias=True)
)
)
(6): Module(
    (input_layernorm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
(self_attention): Module(
    (query_key_value): Linear(in_features=1024, out_features=3072, bias=True)
(attention_dropout): Dropout(p=0.0, inplace=False)
(dense): Linear(in_features=1024, out_features=1024, bias=True)
)
(post_attention_layernorm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
(mlp): Module(
    (dense_h_to_4h): Linear(in_features=1024, out_features=4096, bias=True)
(dense_4h_to_h): Linear(in_features=4096, out_features=1024, bias=True)
)
)
(7): Module(
    (input_layernorm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
(self_attention): Module(
    (query_key_value): Linear(in_features=1024, out_features=3072, bias=True)
(attention_dropout): Dropout(p=0.0, inplace=False)
(dense): Linear(in_features=1024, out_features=1024, bias=True)
)
(post_attention_layernorm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
(mlp): Module(
    (dense_h_to_4h): Linear(in_features=1024, out_features=4096, bias=True)
(dense_4h_to_h): Linear(in_features=4096, out_features=1024, bias=True)
)
)
(8): Module(
    (input_layernorm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
(self_attention): Module(
    (query_key_value): Linear(in_features=1024, out_features=3072, bias=True)
(attention_dropout): Dropout(p=0.0, inplace=False)
(dense): Linear(in_features=1024, out_features=1024, bias=True)
)
(post_attention_layernorm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
(mlp): Module(
    (dense_h_to_4h): Linear(in_features=1024, out_features=4096, bias=True)
(dense_4h_to_h): Linear(in_features=4096, out_features=1024, bias=True)
)
)
(9): Module(
    (input_layernorm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
(self_attention): Module(
    (query_key_value): Linear(in_features=1024, out_features=3072, bias=True)
(attention_dropout): Dropout(p=0.0, inplace=False)
(dense): Linear(in_features=1024, out_features=1024, bias=True)
)
(post_attention_layernorm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
(mlp): Module(
    (dense_h_to_4h): Linear(in_features=1024, out_features=4096, bias=True)
(dense_4h_to_h): Linear(in_features=4096, out_features=1024, bias=True)
)
)
(10): Module(
    (input_layernorm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
(self_attention): Module(
    (query_key_value): Linear(in_features=1024, out_features=3072, bias=True)
(attention_dropout): Dropout(p=0.0, inplace=False)
(dense): Linear(in_features=1024, out_features=1024, bias=True)
)
(post_attention_layernorm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
(mlp): Module(
    (dense_h_to_4h): Linear(in_features=1024, out_features=4096, bias=True)
(dense_4h_to_h): Linear(in_features=4096, out_features=1024, bias=True)
)
)
(11): Module(
    (input_layernorm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
(self_attention): Module(
    (query_key_value): Linear(in_features=1024, out_features=3072, bias=True)
(attention_dropout): Dropout(p=0.0, inplace=False)
(dense): Linear(in_features=1024, out_features=1024, bias=True)
)
(post_attention_layernorm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
(mlp): Module(
    (dense_h_to_4h): Linear(in_features=1024, out_features=4096, bias=True)
(dense_4h_to_h): Linear(in_features=4096, out_features=1024, bias=True)
)
)
(12): Module(
    (input_layernorm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
(self_attention): Module(
    (query_key_value): Linear(in_features=1024, out_features=3072, bias=True)
(attention_dropout): Dropout(p=0.0, inplace=False)
(dense): Linear(in_features=1024, out_features=1024, bias=True)
)
(post_attention_layernorm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
(mlp): Module(
    (dense_h_to_4h): Linear(in_features=1024, out_features=4096, bias=True)
(dense_4h_to_h): Linear(in_features=4096, out_features=1024, bias=True)
)
)
(13): Module(
    (input_layernorm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
(self_attention): Module(
    (query_key_value): Linear(in_features=1024, out_features=3072, bias=True)
(attention_dropout): Dropout(p=0.0, inplace=False)
(dense): Linear(in_features=1024, out_features=1024, bias=True)
)
(post_attention_layernorm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
(mlp): Module(
    (dense_h_to_4h): Linear(in_features=1024, out_features=4096, bias=True)
(dense_4h_to_h): Linear(in_features=4096, out_features=1024, bias=True)
)
)
(14): Module(
    (input_layernorm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
(self_attention): Module(
    (query_key_value): Linear(in_features=1024, out_features=3072, bias=True)
(attention_dropout): Dropout(p=0.0, inplace=False)
(dense): Linear(in_features=1024, out_features=1024, bias=True)
)
(post_attention_layernorm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
(mlp): Module(
    (dense_h_to_4h): Linear(in_features=1024, out_features=4096, bias=True)
(dense_4h_to_h): Linear(in_features=4096, out_features=1024, bias=True)
)
)
(15): Module(
    (input_layernorm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
(self_attention): Module(
    (query_key_value): Linear(in_features=1024, out_features=3072, bias=True)
(attention_dropout): Dropout(p=0.0, inplace=False)
(dense): Linear(in_features=1024, out_features=1024, bias=True)
)
(post_attention_layernorm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
(mlp): Module(
    (dense_h_to_4h): Linear(in_features=1024, out_features=4096, bias=True)
(dense_4h_to_h): Linear(in_features=4096, out_features=1024, bias=True)
)
)
(16): Module(
    (input_layernorm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
(self_attention): Module(
    (query_key_value): Linear(in_features=1024, out_features=3072, bias=True)
(attention_dropout): Dropout(p=0.0, inplace=False)
(dense): Linear(in_features=1024, out_features=1024, bias=True)
)
(post_attention_layernorm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
(mlp): Module(
    (dense_h_to_4h): Linear(in_features=1024, out_features=4096, bias=True)
(dense_4h_to_h): Linear(in_features=4096, out_features=1024, bias=True)
)
)
(17): Module(
    (input_layernorm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
(self_attention): Module(
    (query_key_value): Linear(in_features=1024, out_features=3072, bias=True)
(attention_dropout): Dropout(p=0.0, inplace=False)
(dense): Linear(in_features=1024, out_features=1024, bias=True)
)
(post_attention_layernorm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
(mlp): Module(
    (dense_h_to_4h): Linear(in_features=1024, out_features=4096, bias=True)
(dense_4h_to_h): Linear(in_features=4096, out_features=1024, bias=True)
)
)
(18): Module(
    (input_layernorm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
(self_attention): Module(
    (query_key_value): Linear(in_features=1024, out_features=3072, bias=True)
(attention_dropout): Dropout(p=0.0, inplace=False)
(dense): Linear(in_features=1024, out_features=1024, bias=True)
)
(post_attention_layernorm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
(mlp): Module(
    (dense_h_to_4h): Linear(in_features=1024, out_features=4096, bias=True)
(dense_4h_to_h): Linear(in_features=4096, out_features=1024, bias=True)
)
)
(19): Module(
    (input_layernorm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
(self_attention): Module(
    (query_key_value): Linear(in_features=1024, out_features=3072, bias=True)
(attention_dropout): Dropout(p=0.0, inplace=False)
(dense): Linear(in_features=1024, out_features=1024, bias=True)
)
(post_attention_layernorm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
(mlp): Module(
    (dense_h_to_4h): Linear(in_features=1024, out_features=4096, bias=True)
(dense_4h_to_h): Linear(in_features=4096, out_features=1024, bias=True)
)
)
(20): Module(
    (input_layernorm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
(self_attention): Module(
    (query_key_value): Linear(in_features=1024, out_features=3072, bias=True)
(attention_dropout): Dropout(p=0.0, inplace=False)
(dense): Linear(in_features=1024, out_features=1024, bias=True)
)
(post_attention_layernorm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
(mlp): Module(
    (dense_h_to_4h): Linear(in_features=1024, out_features=4096, bias=True)
(dense_4h_to_h): Linear(in_features=4096, out_features=1024, bias=True)
)
)
(21): Module(
    (input_layernorm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
(self_attention): Module(
    (query_key_value): Linear(in_features=1024, out_features=3072, bias=True)
(attention_dropout): Dropout(p=0.0, inplace=False)
(dense): Linear(in_features=1024, out_features=1024, bias=True)
)
(post_attention_layernorm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
(mlp): Module(
    (dense_h_to_4h): Linear(in_features=1024, out_features=4096, bias=True)
(dense_4h_to_h): Linear(in_features=4096, out_features=1024, bias=True)
)
)
(22): Module(
    (input_layernorm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
(self_attention): Module(
    (query_key_value): Linear(in_features=1024, out_features=3072, bias=True)
(attention_dropout): Dropout(p=0.0, inplace=False)
(dense): Linear(in_features=1024, out_features=1024, bias=True)
)
(post_attention_layernorm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
(mlp): Module(
    (dense_h_to_4h): Linear(in_features=1024, out_features=4096, bias=True)
(dense_4h_to_h): Linear(in_features=4096, out_features=1024, bias=True)
)
)
(23): Module(
    (input_layernorm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
(self_attention): Module(
    (query_key_value): Linear(in_features=1024, out_features=3072, bias=True)
(attention_dropout): Dropout(p=0.0, inplace=False)
(dense): Linear(in_features=1024, out_features=1024, bias=True)
)
(post_attention_layernorm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
(mlp): Module(
    (dense_h_to_4h): Linear(in_features=1024, out_features=4096, bias=True)
(dense_4h_to_h): Linear(in_features=4096, out_features=1024, bias=True)
)
)
)
(ln_f): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
)
(lm_head): Linear(in_features=1024, out_features=250880, bias=False)
)

def forward(self, input_ids: torch.Tensor):
    size = input_ids.size()
    getitem = size[0]
    getitem_1 = size[1];
    size = None
    transformer_word_embeddings = self.transformer.word_embeddings(input_ids);
    input_ids = None
    transformer_word_embeddings_layernorm = self.transformer.word_embeddings_layernorm(transformer_word_embeddings)
    getattr_1 = transformer_word_embeddings_layernorm.device
    ones = torch.ones((getitem, getitem_1), device=getattr_1);
    getitem = getattr_1 = None
    size_1 = ones.size()
    getitem_2 = size_1[0]
    getitem_3 = size_1[1];
    size_1 = None
    getattr_2 = ones.device
    tensor = torch.tensor(0.7071067811865476, device=getattr_2, dtype=torch.float32);
    getattr_2 = None
    getattr_3 = ones.device
    arange = torch.arange(1, 17, device=getattr_3, dtype=torch.int32);
    getattr_3 = None
    pow_1 = torch.pow(tensor, arange);
    tensor = arange = None
    cumsum = ones.cumsum(dim=-1)
    sub = cumsum - 1;
    cumsum = None
    mul = sub * ones;
    sub = None
    getitem_4 = mul[(slice(None, None, None), None, slice(None, None, None))];
    mul = None
    getitem_5 = pow_1[(Ellipsis, None)];
    pow_1 = None
    mul_1 = getitem_5 * getitem_4;
    getitem_5 = getitem_4 = None
    mul_2 = getitem_2 * 16;
    getitem_2 = None
    reshape = mul_1.reshape(mul_2, 1, getitem_3);
    mul_1 = mul_2 = getitem_3 = None
    getattr_4 = transformer_word_embeddings_layernorm.dtype
    to = reshape.to(getattr_4);
    reshape = getattr_4 = None
    add = getitem_1 + 0
    size_2 = ones.size()
    size_3 = ones.size()
    getitem_6 = size_3[0];
    size_3 = None
    gt = getitem_1 > 1
    sub_1 = add - getitem_1;
    add = None
    getattr_5 = transformer_word_embeddings.dtype;
    transformer_word_embeddings = None
    finfo = torch.finfo(getattr_5)
    getattr_6 = finfo.min;
    finfo = None
    getattr_7 = ones.device
    full = torch.full((getitem_1, getitem_1), getattr_6, device=getattr_7);
    getattr_6 = None
    size_4 = full.size(-1)
    arange_1 = torch.arange(size_4, device=getattr_7);
    size_4 = getattr_7 = None
    add_1 = arange_1 + 1
    size_5 = full.size(-1)
    view = add_1.view(size_5, 1);
    add_1 = size_5 = None
    lt = arange_1 < view;
    arange_1 = view = None
    masked_fill_ = full.masked_fill_(lt, 0);
    lt = None
    to_1 = full.to(getattr_5);
    full = None
    gt_1 = sub_1 > 0
    getitem_7 = to_1[(None, None, slice(None, None, None), slice(None, None, None))];
    to_1 = None
    add_2 = getitem_1 + sub_1;
    sub_1 = None
    expand = getitem_7.expand(getitem_6, 1, getitem_1, add_2);
    getitem_7 = getitem_6 = add_2 = None
    size_6 = ones.size()
    getitem_8 = size_6[0]
    getitem_9 = size_6[1];
    size_6 = None
    getitem_10 = ones[(slice(None, None, None), None, None, slice(None, None, None))]
    expand_1 = getitem_10.expand(getitem_8, 1, getitem_1, getitem_9);
    getitem_10 = getitem_8 = getitem_1 = getitem_9 = None
    to_2 = expand_1.to(getattr_5);
    expand_1 = None
    sub_2 = 1.0 - to_2;
    to_2 = None
    to_3 = sub_2.to(torch.bool)
    finfo_1 = torch.finfo(getattr_5)
    getattr_8 = finfo_1.min;
    finfo_1 = None
    masked_fill = sub_2.masked_fill(to_3, getattr_8);
    sub_2 = to_3 = getattr_8 = None
    getattr_9 = ones.device;
    ones = None
    to_4 = masked_fill.to(getattr_9);
    masked_fill = getattr_9 = None
    bool_1 = to_4.bool();
    to_4 = None
    finfo_2 = torch.finfo(getattr_5);
    getattr_5 = None
    getattr_10 = finfo_2.min;
    finfo_2 = None
    masked_fill_1 = expand.masked_fill(bool_1, getattr_10);
    expand = bool_1 = getattr_10 = None
    bool_2 = masked_fill_1.bool();
    masked_fill_1 = None
    transformer_h_0_input_layernorm = getattr(self.transformer.h, "0").input_layernorm(
        transformer_word_embeddings_layernorm)
    transformer_h_0_self_attention_query_key_value = getattr(self.transformer.h, "0").self_attention.query_key_value(
        transformer_h_0_input_layernorm);
    transformer_h_0_input_layernorm = None
    size_7 = transformer_h_0_self_attention_query_key_value.size()
    getitem_11 = size_7[0]
    getitem_12 = size_7[1]
    getitem_13 = size_7[2];
    size_7 = None
    view_1 = transformer_h_0_self_attention_query_key_value.view(getitem_11, getitem_12, 16, 3, 64);
    transformer_h_0_self_attention_query_key_value = getitem_11 = getitem_12 = None
    getitem_14 = view_1[(Ellipsis, 0, slice(None, None, None))]
    getitem_15 = view_1[(Ellipsis, 1, slice(None, None, None))]
    getitem_16 = view_1[(Ellipsis, 2, slice(None, None, None))];
    view_1 = None
    size_8 = getitem_14.size()
    getitem_17 = size_8[0]
    getitem_18 = size_8[1]
    getitem_19 = size_8[2]
    getitem_20 = size_8[3];
    size_8 = None
    transpose = getitem_14.transpose(1, 2);
    getitem_14 = None
    mul_3 = getitem_17 * 16
    reshape_1 = transpose.reshape(mul_3, getitem_18, 64);
    transpose = mul_3 = None
    permute = getitem_15.permute(0, 2, 3, 1);
    getitem_15 = None
    mul_4 = getitem_17 * 16
    reshape_2 = permute.reshape(mul_4, 64, getitem_18);
    permute = mul_4 = None
    transpose_1 = getitem_16.transpose(1, 2);
    getitem_16 = None
    mul_5 = getitem_17 * 16
    reshape_3 = transpose_1.reshape(mul_5, getitem_18, 64);
    transpose_1 = mul_5 = None
    size_9 = reshape_2.size()
    getitem_21 = size_9[0]
    getitem_22 = size_9[1]
    getitem_23 = size_9[2];
    size_9 = None
    baddbmm = to.baddbmm(batch1=reshape_1, batch2=reshape_2, beta=1.0, alpha=0.125);
    reshape_1 = reshape_2 = None
    view_2 = baddbmm.view(getitem_17, 16, getitem_18, getitem_23);
    baddbmm = None
    getattr_11 = view_2.dtype
    eq = getattr_11 == torch.float16
    getattr_12 = view_2.dtype
    finfo_3 = torch.finfo(getattr_12);
    getattr_12 = None
    getattr_13 = finfo_3.min;
    finfo_3 = None
    masked_fill_2 = torch.masked_fill(view_2, bool_2, getattr_13);
    view_2 = getattr_13 = None
    softmax = torch.nn.functional.softmax(masked_fill_2, dim=-1, _stacklevel=3, dtype=torch.float32);
    masked_fill_2 = None
    to_5 = softmax.to(getattr_11);
    softmax = getattr_11 = None
    transformer_h_0_self_attention_attention_dropout = getattr(self.transformer.h,
                                                               "0").self_attention.attention_dropout(to_5);
    to_5 = None
    mul_6 = getitem_17 * 16;
    getitem_17 = None
    view_3 = transformer_h_0_self_attention_attention_dropout.view(mul_6, getitem_18, getitem_23);
    transformer_h_0_self_attention_attention_dropout = mul_6 = getitem_18 = getitem_23 = None
    bmm = torch.bmm(view_3, reshape_3);
    view_3 = reshape_3 = None
    size_10 = bmm.size()
    getitem_24 = size_10[0]
    getitem_25 = size_10[1]
    getitem_26 = size_10[2];
    size_10 = None
    floordiv = getitem_24 // 16;
    getitem_24 = None
    view_4 = bmm.view(floordiv, 16, getitem_25, 64);
    bmm = None
    permute_1 = view_4.permute(0, 2, 1, 3);
    view_4 = None
    reshape_4 = permute_1.reshape(floordiv, getitem_25, 1024);
    permute_1 = floordiv = getitem_25 = None
    transformer_h_0_self_attention_dense = getattr(self.transformer.h, "0").self_attention.dense(reshape_4);
    reshape_4 = None
    dropout = torch.nn.functional.dropout(transformer_h_0_self_attention_dense, p=0.0, training=False, inplace=False);
    transformer_h_0_self_attention_dense = None
    add_3 = transformer_word_embeddings_layernorm + dropout;
    transformer_word_embeddings_layernorm = dropout = None
    transformer_h_0_post_attention_layernorm = getattr(self.transformer.h, "0").post_attention_layernorm(add_3)
    transformer_h_0_mlp_dense_h_to_4h = getattr(self.transformer.h, "0").mlp.dense_h_to_4h(
        transformer_h_0_post_attention_layernorm);
    transformer_h_0_post_attention_layernorm = None
    mul_7 = transformer_h_0_mlp_dense_h_to_4h * 0.5
    mul_8 = 0.79788456 * transformer_h_0_mlp_dense_h_to_4h
    mul_9 = 0.044715 * transformer_h_0_mlp_dense_h_to_4h
    mul_10 = mul_9 * transformer_h_0_mlp_dense_h_to_4h;
    mul_9 = transformer_h_0_mlp_dense_h_to_4h = None
    add_4 = 1 + mul_10;
    mul_10 = None
    mul_11 = mul_8 * add_4;
    mul_8 = add_4 = None
    tanh = torch.tanh(mul_11);
    mul_11 = None
    add_5 = 1.0 + tanh;
    tanh = None
    mul_12 = mul_7 * add_5;
    mul_7 = add_5 = None
    transformer_h_0_mlp_dense_4h_to_h = getattr(self.transformer.h, "0").mlp.dense_4h_to_h(mul_12);
    mul_12 = None
    dropout_1 = torch.nn.functional.dropout(transformer_h_0_mlp_dense_4h_to_h, p=0.0, training=False, inplace=False);
    transformer_h_0_mlp_dense_4h_to_h = None
    add_6 = add_3 + dropout_1;
    add_3 = dropout_1 = None


    transformer_h_1_input_layernorm = getattr(self.transformer.h, "1").input_layernorm(add_6)
    transformer_h_1_self_attention_query_key_value = getattr(self.transformer.h, "1").self_attention.query_key_value(
        transformer_h_1_input_layernorm);
    transformer_h_1_input_layernorm = None
    size_11 = transformer_h_1_self_attention_query_key_value.size()
    getitem_27 = size_11[0]
    getitem_28 = size_11[1]
    getitem_29 = size_11[2];
    size_11 = None
    view_5 = transformer_h_1_self_attention_query_key_value.view(getitem_27, getitem_28, 16, 3, 64);
    transformer_h_1_self_attention_query_key_value = getitem_27 = getitem_28 = None
    getitem_30 = view_5[(Ellipsis, 0, slice(None, None, None))]
    getitem_31 = view_5[(Ellipsis, 1, slice(None, None, None))]
    getitem_32 = view_5[(Ellipsis, 2, slice(None, None, None))];
    view_5 = None
    size_12 = getitem_30.size()
    getitem_33 = size_12[0]
    getitem_34 = size_12[1]
    getitem_35 = size_12[2]
    getitem_36 = size_12[3];
    size_12 = None
    transpose_2 = getitem_30.transpose(1, 2);
    getitem_30 = None
    mul_13 = getitem_33 * 16
    reshape_5 = transpose_2.reshape(mul_13, getitem_34, 64);
    transpose_2 = mul_13 = None
    permute_2 = getitem_31.permute(0, 2, 3, 1);
    getitem_31 = None
    mul_14 = getitem_33 * 16
    reshape_6 = permute_2.reshape(mul_14, 64, getitem_34);
    permute_2 = mul_14 = None
    transpose_3 = getitem_32.transpose(1, 2);
    getitem_32 = None
    mul_15 = getitem_33 * 16
    reshape_7 = transpose_3.reshape(mul_15, getitem_34, 64);
    transpose_3 = mul_15 = None
    size_13 = reshape_6.size()
    getitem_37 = size_13[0]
    getitem_38 = size_13[1]
    getitem_39 = size_13[2];
    size_13 = None
    baddbmm_1 = to.baddbmm(batch1=reshape_5, batch2=reshape_6, beta=1.0, alpha=0.125);
    reshape_5 = reshape_6 = None
    view_6 = baddbmm_1.view(getitem_33, 16, getitem_34, getitem_39);
    baddbmm_1 = None
    getattr_14 = view_6.dtype
    eq_1 = getattr_14 == torch.float16
    getattr_15 = view_6.dtype
    finfo_4 = torch.finfo(getattr_15);
    getattr_15 = None
    getattr_16 = finfo_4.min;
    finfo_4 = None
    masked_fill_3 = torch.masked_fill(view_6, bool_2, getattr_16);
    view_6 = getattr_16 = None
    softmax_1 = torch.nn.functional.softmax(masked_fill_3, dim=-1, _stacklevel=3, dtype=torch.float32);
    masked_fill_3 = None
    to_6 = softmax_1.to(getattr_14);
    softmax_1 = getattr_14 = None
    transformer_h_1_self_attention_attention_dropout = getattr(self.transformer.h,
                                                               "1").self_attention.attention_dropout(to_6);
    to_6 = None
    mul_16 = getitem_33 * 16;
    getitem_33 = None
    view_7 = transformer_h_1_self_attention_attention_dropout.view(mul_16, getitem_34, getitem_39);
    transformer_h_1_self_attention_attention_dropout = mul_16 = getitem_34 = getitem_39 = None
    bmm_1 = torch.bmm(view_7, reshape_7);
    view_7 = reshape_7 = None
    size_14 = bmm_1.size()
    getitem_40 = size_14[0]
    getitem_41 = size_14[1]
    getitem_42 = size_14[2];
    size_14 = None
    floordiv_1 = getitem_40 // 16;
    getitem_40 = None
    view_8 = bmm_1.view(floordiv_1, 16, getitem_41, 64);
    bmm_1 = None
    permute_3 = view_8.permute(0, 2, 1, 3);
    view_8 = None
    reshape_8 = permute_3.reshape(floordiv_1, getitem_41, 1024);
    permute_3 = floordiv_1 = getitem_41 = None
    transformer_h_1_self_attention_dense = getattr(self.transformer.h, "1").self_attention.dense(reshape_8);
    reshape_8 = None
    dropout_2 = torch.nn.functional.dropout(transformer_h_1_self_attention_dense, p=0.0, training=False, inplace=False);
    transformer_h_1_self_attention_dense = None
    add_7 = add_6 + dropout_2;
    add_6 = dropout_2 = None
    transformer_h_1_post_attention_layernorm = getattr(self.transformer.h, "1").post_attention_layernorm(add_7)
    transformer_h_1_mlp_dense_h_to_4h = getattr(self.transformer.h, "1").mlp.dense_h_to_4h(
        transformer_h_1_post_attention_layernorm);
    transformer_h_1_post_attention_layernorm = None
    mul_17 = transformer_h_1_mlp_dense_h_to_4h * 0.5
    mul_18 = 0.79788456 * transformer_h_1_mlp_dense_h_to_4h
    mul_19 = 0.044715 * transformer_h_1_mlp_dense_h_to_4h
    mul_20 = mul_19 * transformer_h_1_mlp_dense_h_to_4h;
    mul_19 = transformer_h_1_mlp_dense_h_to_4h = None
    add_8 = 1 + mul_20;
    mul_20 = None
    mul_21 = mul_18 * add_8;
    mul_18 = add_8 = None
    tanh_1 = torch.tanh(mul_21);
    mul_21 = None
    add_9 = 1.0 + tanh_1;
    tanh_1 = None
    mul_22 = mul_17 * add_9;
    mul_17 = add_9 = None
    transformer_h_1_mlp_dense_4h_to_h = getattr(self.transformer.h, "1").mlp.dense_4h_to_h(mul_22);
    mul_22 = None
    dropout_3 = torch.nn.functional.dropout(transformer_h_1_mlp_dense_4h_to_h, p=0.0, training=False, inplace=False);
    transformer_h_1_mlp_dense_4h_to_h = None
    add_10 = add_7 + dropout_3;
    add_7 = dropout_3 = None
    transformer_h_2_input_layernorm = getattr(self.transformer.h, "2").input_layernorm(add_10)
    transformer_h_2_self_attention_query_key_value = getattr(self.transformer.h, "2").self_attention.query_key_value(
        transformer_h_2_input_layernorm);
    transformer_h_2_input_layernorm = None
    size_15 = transformer_h_2_self_attention_query_key_value.size()
    getitem_43 = size_15[0]
    getitem_44 = size_15[1]
    getitem_45 = size_15[2];
    size_15 = None
    view_9 = transformer_h_2_self_attention_query_key_value.view(getitem_43, getitem_44, 16, 3, 64);
    transformer_h_2_self_attention_query_key_value = getitem_43 = getitem_44 = None
    getitem_46 = view_9[(Ellipsis, 0, slice(None, None, None))]
    getitem_47 = view_9[(Ellipsis, 1, slice(None, None, None))]
    getitem_48 = view_9[(Ellipsis, 2, slice(None, None, None))];
    view_9 = None
    size_16 = getitem_46.size()
    getitem_49 = size_16[0]
    getitem_50 = size_16[1]
    getitem_51 = size_16[2]
    getitem_52 = size_16[3];
    size_16 = None
    transpose_4 = getitem_46.transpose(1, 2);
    getitem_46 = None
    mul_23 = getitem_49 * 16
    reshape_9 = transpose_4.reshape(mul_23, getitem_50, 64);
    transpose_4 = mul_23 = None
    permute_4 = getitem_47.permute(0, 2, 3, 1);
    getitem_47 = None
    mul_24 = getitem_49 * 16
    reshape_10 = permute_4.reshape(mul_24, 64, getitem_50);
    permute_4 = mul_24 = None
    transpose_5 = getitem_48.transpose(1, 2);
    getitem_48 = None
    mul_25 = getitem_49 * 16
    reshape_11 = transpose_5.reshape(mul_25, getitem_50, 64);
    transpose_5 = mul_25 = None
    size_17 = reshape_10.size()
    getitem_53 = size_17[0]
    getitem_54 = size_17[1]
    getitem_55 = size_17[2];
    size_17 = None
    baddbmm_2 = to.baddbmm(batch1=reshape_9, batch2=reshape_10, beta=1.0, alpha=0.125);
    reshape_9 = reshape_10 = None
    view_10 = baddbmm_2.view(getitem_49, 16, getitem_50, getitem_55);
    baddbmm_2 = None
    getattr_17 = view_10.dtype
    eq_2 = getattr_17 == torch.float16
    getattr_18 = view_10.dtype
    finfo_5 = torch.finfo(getattr_18);
    getattr_18 = None
    getattr_19 = finfo_5.min;
    finfo_5 = None
    masked_fill_4 = torch.masked_fill(view_10, bool_2, getattr_19);
    view_10 = getattr_19 = None
    softmax_2 = torch.nn.functional.softmax(masked_fill_4, dim=-1, _stacklevel=3, dtype=torch.float32);
    masked_fill_4 = None
    to_7 = softmax_2.to(getattr_17);
    softmax_2 = getattr_17 = None
    transformer_h_2_self_attention_attention_dropout = getattr(self.transformer.h,
                                                               "2").self_attention.attention_dropout(to_7);
    to_7 = None
    mul_26 = getitem_49 * 16;
    getitem_49 = None
    view_11 = transformer_h_2_self_attention_attention_dropout.view(mul_26, getitem_50, getitem_55);
    transformer_h_2_self_attention_attention_dropout = mul_26 = getitem_50 = getitem_55 = None
    bmm_2 = torch.bmm(view_11, reshape_11);
    view_11 = reshape_11 = None
    size_18 = bmm_2.size()
    getitem_56 = size_18[0]
    getitem_57 = size_18[1]
    getitem_58 = size_18[2];
    size_18 = None
    floordiv_2 = getitem_56 // 16;
    getitem_56 = None
    view_12 = bmm_2.view(floordiv_2, 16, getitem_57, 64);
    bmm_2 = None
    permute_5 = view_12.permute(0, 2, 1, 3);
    view_12 = None
    reshape_12 = permute_5.reshape(floordiv_2, getitem_57, 1024);
    permute_5 = floordiv_2 = getitem_57 = None
    transformer_h_2_self_attention_dense = getattr(self.transformer.h, "2").self_attention.dense(reshape_12);
    reshape_12 = None
    dropout_4 = torch.nn.functional.dropout(transformer_h_2_self_attention_dense, p=0.0, training=False, inplace=False);
    transformer_h_2_self_attention_dense = None
    add_11 = add_10 + dropout_4;
    add_10 = dropout_4 = None
    transformer_h_2_post_attention_layernorm = getattr(self.transformer.h, "2").post_attention_layernorm(add_11)
    transformer_h_2_mlp_dense_h_to_4h = getattr(self.transformer.h, "2").mlp.dense_h_to_4h(
        transformer_h_2_post_attention_layernorm);
    transformer_h_2_post_attention_layernorm = None
    mul_27 = transformer_h_2_mlp_dense_h_to_4h * 0.5
    mul_28 = 0.79788456 * transformer_h_2_mlp_dense_h_to_4h
    mul_29 = 0.044715 * transformer_h_2_mlp_dense_h_to_4h
    mul_30 = mul_29 * transformer_h_2_mlp_dense_h_to_4h;
    mul_29 = transformer_h_2_mlp_dense_h_to_4h = None
    add_12 = 1 + mul_30;
    mul_30 = None
    mul_31 = mul_28 * add_12;
    mul_28 = add_12 = None
    tanh_2 = torch.tanh(mul_31);
    mul_31 = None
    add_13 = 1.0 + tanh_2;
    tanh_2 = None
    mul_32 = mul_27 * add_13;
    mul_27 = add_13 = None
    transformer_h_2_mlp_dense_4h_to_h = getattr(self.transformer.h, "2").mlp.dense_4h_to_h(mul_32);
    mul_32 = None
    dropout_5 = torch.nn.functional.dropout(transformer_h_2_mlp_dense_4h_to_h, p=0.0, training=False, inplace=False);
    transformer_h_2_mlp_dense_4h_to_h = None
    add_14 = add_11 + dropout_5;
    add_11 = dropout_5 = None
    transformer_h_3_input_layernorm = getattr(self.transformer.h, "3").input_layernorm(add_14)
    transformer_h_3_self_attention_query_key_value = getattr(self.transformer.h, "3").self_attention.query_key_value(
        transformer_h_3_input_layernorm);
    transformer_h_3_input_layernorm = None
    size_19 = transformer_h_3_self_attention_query_key_value.size()
    getitem_59 = size_19[0]
    getitem_60 = size_19[1]
    getitem_61 = size_19[2];
    size_19 = None
    view_13 = transformer_h_3_self_attention_query_key_value.view(getitem_59, getitem_60, 16, 3, 64);
    transformer_h_3_self_attention_query_key_value = getitem_59 = getitem_60 = None
    getitem_62 = view_13[(Ellipsis, 0, slice(None, None, None))]
    getitem_63 = view_13[(Ellipsis, 1, slice(None, None, None))]
    getitem_64 = view_13[(Ellipsis, 2, slice(None, None, None))];
    view_13 = None
    size_20 = getitem_62.size()
    getitem_65 = size_20[0]
    getitem_66 = size_20[1]
    getitem_67 = size_20[2]
    getitem_68 = size_20[3];
    size_20 = None
    transpose_6 = getitem_62.transpose(1, 2);
    getitem_62 = None
    mul_33 = getitem_65 * 16
    reshape_13 = transpose_6.reshape(mul_33, getitem_66, 64);
    transpose_6 = mul_33 = None
    permute_6 = getitem_63.permute(0, 2, 3, 1);
    getitem_63 = None
    mul_34 = getitem_65 * 16
    reshape_14 = permute_6.reshape(mul_34, 64, getitem_66);
    permute_6 = mul_34 = None
    transpose_7 = getitem_64.transpose(1, 2);
    getitem_64 = None
    mul_35 = getitem_65 * 16
    reshape_15 = transpose_7.reshape(mul_35, getitem_66, 64);
    transpose_7 = mul_35 = None
    size_21 = reshape_14.size()
    getitem_69 = size_21[0]
    getitem_70 = size_21[1]
    getitem_71 = size_21[2];
    size_21 = None
    baddbmm_3 = to.baddbmm(batch1=reshape_13, batch2=reshape_14, beta=1.0, alpha=0.125);
    reshape_13 = reshape_14 = None
    view_14 = baddbmm_3.view(getitem_65, 16, getitem_66, getitem_71);
    baddbmm_3 = None
    getattr_20 = view_14.dtype
    eq_3 = getattr_20 == torch.float16
    getattr_21 = view_14.dtype
    finfo_6 = torch.finfo(getattr_21);
    getattr_21 = None
    getattr_22 = finfo_6.min;
    finfo_6 = None
    masked_fill_5 = torch.masked_fill(view_14, bool_2, getattr_22);
    view_14 = getattr_22 = None
    softmax_3 = torch.nn.functional.softmax(masked_fill_5, dim=-1, _stacklevel=3, dtype=torch.float32);
    masked_fill_5 = None
    to_8 = softmax_3.to(getattr_20);
    softmax_3 = getattr_20 = None
    transformer_h_3_self_attention_attention_dropout = getattr(self.transformer.h,
                                                               "3").self_attention.attention_dropout(to_8);
    to_8 = None
    mul_36 = getitem_65 * 16;
    getitem_65 = None
    view_15 = transformer_h_3_self_attention_attention_dropout.view(mul_36, getitem_66, getitem_71);
    transformer_h_3_self_attention_attention_dropout = mul_36 = getitem_66 = getitem_71 = None
    bmm_3 = torch.bmm(view_15, reshape_15);
    view_15 = reshape_15 = None
    size_22 = bmm_3.size()
    getitem_72 = size_22[0]
    getitem_73 = size_22[1]
    getitem_74 = size_22[2];
    size_22 = None
    floordiv_3 = getitem_72 // 16;
    getitem_72 = None
    view_16 = bmm_3.view(floordiv_3, 16, getitem_73, 64);
    bmm_3 = None
    permute_7 = view_16.permute(0, 2, 1, 3);
    view_16 = None
    reshape_16 = permute_7.reshape(floordiv_3, getitem_73, 1024);
    permute_7 = floordiv_3 = getitem_73 = None
    transformer_h_3_self_attention_dense = getattr(self.transformer.h, "3").self_attention.dense(reshape_16);
    reshape_16 = None
    dropout_6 = torch.nn.functional.dropout(transformer_h_3_self_attention_dense, p=0.0, training=False, inplace=False);
    transformer_h_3_self_attention_dense = None
    add_15 = add_14 + dropout_6;
    add_14 = dropout_6 = None
    transformer_h_3_post_attention_layernorm = getattr(self.transformer.h, "3").post_attention_layernorm(add_15)
    transformer_h_3_mlp_dense_h_to_4h = getattr(self.transformer.h, "3").mlp.dense_h_to_4h(
        transformer_h_3_post_attention_layernorm);
    transformer_h_3_post_attention_layernorm = None
    mul_37 = transformer_h_3_mlp_dense_h_to_4h * 0.5
    mul_38 = 0.79788456 * transformer_h_3_mlp_dense_h_to_4h
    mul_39 = 0.044715 * transformer_h_3_mlp_dense_h_to_4h
    mul_40 = mul_39 * transformer_h_3_mlp_dense_h_to_4h;
    mul_39 = transformer_h_3_mlp_dense_h_to_4h = None
    add_16 = 1 + mul_40;
    mul_40 = None
    mul_41 = mul_38 * add_16;
    mul_38 = add_16 = None
    tanh_3 = torch.tanh(mul_41);
    mul_41 = None
    add_17 = 1.0 + tanh_3;
    tanh_3 = None
    mul_42 = mul_37 * add_17;
    mul_37 = add_17 = None
    transformer_h_3_mlp_dense_4h_to_h = getattr(self.transformer.h, "3").mlp.dense_4h_to_h(mul_42);
    mul_42 = None
    dropout_7 = torch.nn.functional.dropout(transformer_h_3_mlp_dense_4h_to_h, p=0.0, training=False, inplace=False);
    transformer_h_3_mlp_dense_4h_to_h = None
    add_18 = add_15 + dropout_7;
    add_15 = dropout_7 = None
    transformer_h_4_input_layernorm = getattr(self.transformer.h, "4").input_layernorm(add_18)
    transformer_h_4_self_attention_query_key_value = getattr(self.transformer.h, "4").self_attention.query_key_value(
        transformer_h_4_input_layernorm);
    transformer_h_4_input_layernorm = None
    size_23 = transformer_h_4_self_attention_query_key_value.size()
    getitem_75 = size_23[0]
    getitem_76 = size_23[1]
    getitem_77 = size_23[2];
    size_23 = None
    view_17 = transformer_h_4_self_attention_query_key_value.view(getitem_75, getitem_76, 16, 3, 64);
    transformer_h_4_self_attention_query_key_value = getitem_75 = getitem_76 = None
    getitem_78 = view_17[(Ellipsis, 0, slice(None, None, None))]
    getitem_79 = view_17[(Ellipsis, 1, slice(None, None, None))]
    getitem_80 = view_17[(Ellipsis, 2, slice(None, None, None))];
    view_17 = None
    size_24 = getitem_78.size()
    getitem_81 = size_24[0]
    getitem_82 = size_24[1]
    getitem_83 = size_24[2]
    getitem_84 = size_24[3];
    size_24 = None
    transpose_8 = getitem_78.transpose(1, 2);
    getitem_78 = None
    mul_43 = getitem_81 * 16
    reshape_17 = transpose_8.reshape(mul_43, getitem_82, 64);
    transpose_8 = mul_43 = None
    permute_8 = getitem_79.permute(0, 2, 3, 1);
    getitem_79 = None
    mul_44 = getitem_81 * 16
    reshape_18 = permute_8.reshape(mul_44, 64, getitem_82);
    permute_8 = mul_44 = None
    transpose_9 = getitem_80.transpose(1, 2);
    getitem_80 = None
    mul_45 = getitem_81 * 16
    reshape_19 = transpose_9.reshape(mul_45, getitem_82, 64);
    transpose_9 = mul_45 = None
    size_25 = reshape_18.size()
    getitem_85 = size_25[0]
    getitem_86 = size_25[1]
    getitem_87 = size_25[2];
    size_25 = None
    baddbmm_4 = to.baddbmm(batch1=reshape_17, batch2=reshape_18, beta=1.0, alpha=0.125);
    reshape_17 = reshape_18 = None
    view_18 = baddbmm_4.view(getitem_81, 16, getitem_82, getitem_87);
    baddbmm_4 = None
    getattr_23 = view_18.dtype
    eq_4 = getattr_23 == torch.float16
    getattr_24 = view_18.dtype
    finfo_7 = torch.finfo(getattr_24);
    getattr_24 = None
    getattr_25 = finfo_7.min;
    finfo_7 = None
    masked_fill_6 = torch.masked_fill(view_18, bool_2, getattr_25);
    view_18 = getattr_25 = None
    softmax_4 = torch.nn.functional.softmax(masked_fill_6, dim=-1, _stacklevel=3, dtype=torch.float32);
    masked_fill_6 = None
    to_9 = softmax_4.to(getattr_23);
    softmax_4 = getattr_23 = None
    transformer_h_4_self_attention_attention_dropout = getattr(self.transformer.h,
                                                               "4").self_attention.attention_dropout(to_9);
    to_9 = None
    mul_46 = getitem_81 * 16;
    getitem_81 = None
    view_19 = transformer_h_4_self_attention_attention_dropout.view(mul_46, getitem_82, getitem_87);
    transformer_h_4_self_attention_attention_dropout = mul_46 = getitem_82 = getitem_87 = None
    bmm_4 = torch.bmm(view_19, reshape_19);
    view_19 = reshape_19 = None
    size_26 = bmm_4.size()
    getitem_88 = size_26[0]
    getitem_89 = size_26[1]
    getitem_90 = size_26[2];
    size_26 = None
    floordiv_4 = getitem_88 // 16;
    getitem_88 = None
    view_20 = bmm_4.view(floordiv_4, 16, getitem_89, 64);
    bmm_4 = None
    permute_9 = view_20.permute(0, 2, 1, 3);
    view_20 = None
    reshape_20 = permute_9.reshape(floordiv_4, getitem_89, 1024);
    permute_9 = floordiv_4 = getitem_89 = None
    transformer_h_4_self_attention_dense = getattr(self.transformer.h, "4").self_attention.dense(reshape_20);
    reshape_20 = None
    dropout_8 = torch.nn.functional.dropout(transformer_h_4_self_attention_dense, p=0.0, training=False, inplace=False);
    transformer_h_4_self_attention_dense = None
    add_19 = add_18 + dropout_8;
    add_18 = dropout_8 = None
    transformer_h_4_post_attention_layernorm = getattr(self.transformer.h, "4").post_attention_layernorm(add_19)
    transformer_h_4_mlp_dense_h_to_4h = getattr(self.transformer.h, "4").mlp.dense_h_to_4h(
        transformer_h_4_post_attention_layernorm);
    transformer_h_4_post_attention_layernorm = None
    mul_47 = transformer_h_4_mlp_dense_h_to_4h * 0.5
    mul_48 = 0.79788456 * transformer_h_4_mlp_dense_h_to_4h
    mul_49 = 0.044715 * transformer_h_4_mlp_dense_h_to_4h
    mul_50 = mul_49 * transformer_h_4_mlp_dense_h_to_4h;
    mul_49 = transformer_h_4_mlp_dense_h_to_4h = None
    add_20 = 1 + mul_50;
    mul_50 = None
    mul_51 = mul_48 * add_20;
    mul_48 = add_20 = None
    tanh_4 = torch.tanh(mul_51);
    mul_51 = None
    add_21 = 1.0 + tanh_4;
    tanh_4 = None
    mul_52 = mul_47 * add_21;
    mul_47 = add_21 = None
    transformer_h_4_mlp_dense_4h_to_h = getattr(self.transformer.h, "4").mlp.dense_4h_to_h(mul_52);
    mul_52 = None
    dropout_9 = torch.nn.functional.dropout(transformer_h_4_mlp_dense_4h_to_h, p=0.0, training=False, inplace=False);
    transformer_h_4_mlp_dense_4h_to_h = None
    add_22 = add_19 + dropout_9;
    add_19 = dropout_9 = None
    transformer_h_5_input_layernorm = getattr(self.transformer.h, "5").input_layernorm(add_22)
    transformer_h_5_self_attention_query_key_value = getattr(self.transformer.h, "5").self_attention.query_key_value(
        transformer_h_5_input_layernorm);
    transformer_h_5_input_layernorm = None
    size_27 = transformer_h_5_self_attention_query_key_value.size()
    getitem_91 = size_27[0]
    getitem_92 = size_27[1]
    getitem_93 = size_27[2];
    size_27 = None
    view_21 = transformer_h_5_self_attention_query_key_value.view(getitem_91, getitem_92, 16, 3, 64);
    transformer_h_5_self_attention_query_key_value = getitem_91 = getitem_92 = None
    getitem_94 = view_21[(Ellipsis, 0, slice(None, None, None))]
    getitem_95 = view_21[(Ellipsis, 1, slice(None, None, None))]
    getitem_96 = view_21[(Ellipsis, 2, slice(None, None, None))];
    view_21 = None
    size_28 = getitem_94.size()
    getitem_97 = size_28[0]
    getitem_98 = size_28[1]
    getitem_99 = size_28[2]
    getitem_100 = size_28[3];
    size_28 = None
    transpose_10 = getitem_94.transpose(1, 2);
    getitem_94 = None
    mul_53 = getitem_97 * 16
    reshape_21 = transpose_10.reshape(mul_53, getitem_98, 64);
    transpose_10 = mul_53 = None
    permute_10 = getitem_95.permute(0, 2, 3, 1);
    getitem_95 = None
    mul_54 = getitem_97 * 16
    reshape_22 = permute_10.reshape(mul_54, 64, getitem_98);
    permute_10 = mul_54 = None
    transpose_11 = getitem_96.transpose(1, 2);
    getitem_96 = None
    mul_55 = getitem_97 * 16
    reshape_23 = transpose_11.reshape(mul_55, getitem_98, 64);
    transpose_11 = mul_55 = None
    size_29 = reshape_22.size()
    getitem_101 = size_29[0]
    getitem_102 = size_29[1]
    getitem_103 = size_29[2];
    size_29 = None
    baddbmm_5 = to.baddbmm(batch1=reshape_21, batch2=reshape_22, beta=1.0, alpha=0.125);
    reshape_21 = reshape_22 = None
    view_22 = baddbmm_5.view(getitem_97, 16, getitem_98, getitem_103);
    baddbmm_5 = None
    getattr_26 = view_22.dtype
    eq_5 = getattr_26 == torch.float16
    getattr_27 = view_22.dtype
    finfo_8 = torch.finfo(getattr_27);
    getattr_27 = None
    getattr_28 = finfo_8.min;
    finfo_8 = None
    masked_fill_7 = torch.masked_fill(view_22, bool_2, getattr_28);
    view_22 = getattr_28 = None
    softmax_5 = torch.nn.functional.softmax(masked_fill_7, dim=-1, _stacklevel=3, dtype=torch.float32);
    masked_fill_7 = None
    to_10 = softmax_5.to(getattr_26);
    softmax_5 = getattr_26 = None
    transformer_h_5_self_attention_attention_dropout = getattr(self.transformer.h,
                                                               "5").self_attention.attention_dropout(to_10);
    to_10 = None
    mul_56 = getitem_97 * 16;
    getitem_97 = None
    view_23 = transformer_h_5_self_attention_attention_dropout.view(mul_56, getitem_98, getitem_103);
    transformer_h_5_self_attention_attention_dropout = mul_56 = getitem_98 = getitem_103 = None
    bmm_5 = torch.bmm(view_23, reshape_23);
    view_23 = reshape_23 = None
    size_30 = bmm_5.size()
    getitem_104 = size_30[0]
    getitem_105 = size_30[1]
    getitem_106 = size_30[2];
    size_30 = None
    floordiv_5 = getitem_104 // 16;
    getitem_104 = None
    view_24 = bmm_5.view(floordiv_5, 16, getitem_105, 64);
    bmm_5 = None
    permute_11 = view_24.permute(0, 2, 1, 3);
    view_24 = None
    reshape_24 = permute_11.reshape(floordiv_5, getitem_105, 1024);
    permute_11 = floordiv_5 = getitem_105 = None
    transformer_h_5_self_attention_dense = getattr(self.transformer.h, "5").self_attention.dense(reshape_24);
    reshape_24 = None
    dropout_10 = torch.nn.functional.dropout(transformer_h_5_self_attention_dense, p=0.0, training=False,
                                             inplace=False);
    transformer_h_5_self_attention_dense = None
    add_23 = add_22 + dropout_10;
    add_22 = dropout_10 = None
    transformer_h_5_post_attention_layernorm = getattr(self.transformer.h, "5").post_attention_layernorm(add_23)
    transformer_h_5_mlp_dense_h_to_4h = getattr(self.transformer.h, "5").mlp.dense_h_to_4h(
        transformer_h_5_post_attention_layernorm);
    transformer_h_5_post_attention_layernorm = None
    mul_57 = transformer_h_5_mlp_dense_h_to_4h * 0.5
    mul_58 = 0.79788456 * transformer_h_5_mlp_dense_h_to_4h
    mul_59 = 0.044715 * transformer_h_5_mlp_dense_h_to_4h
    mul_60 = mul_59 * transformer_h_5_mlp_dense_h_to_4h;
    mul_59 = transformer_h_5_mlp_dense_h_to_4h = None
    add_24 = 1 + mul_60;
    mul_60 = None
    mul_61 = mul_58 * add_24;
    mul_58 = add_24 = None
    tanh_5 = torch.tanh(mul_61);
    mul_61 = None
    add_25 = 1.0 + tanh_5;
    tanh_5 = None
    mul_62 = mul_57 * add_25;
    mul_57 = add_25 = None
    transformer_h_5_mlp_dense_4h_to_h = getattr(self.transformer.h, "5").mlp.dense_4h_to_h(mul_62);
    mul_62 = None
    dropout_11 = torch.nn.functional.dropout(transformer_h_5_mlp_dense_4h_to_h, p=0.0, training=False, inplace=False);
    transformer_h_5_mlp_dense_4h_to_h = None
    add_26 = add_23 + dropout_11;
    add_23 = dropout_11 = None
    transformer_h_6_input_layernorm = getattr(self.transformer.h, "6").input_layernorm(add_26)
    transformer_h_6_self_attention_query_key_value = getattr(self.transformer.h, "6").self_attention.query_key_value(
        transformer_h_6_input_layernorm);
    transformer_h_6_input_layernorm = None
    size_31 = transformer_h_6_self_attention_query_key_value.size()
    getitem_107 = size_31[0]
    getitem_108 = size_31[1]
    getitem_109 = size_31[2];
    size_31 = None
    view_25 = transformer_h_6_self_attention_query_key_value.view(getitem_107, getitem_108, 16, 3, 64);
    transformer_h_6_self_attention_query_key_value = getitem_107 = getitem_108 = None
    getitem_110 = view_25[(Ellipsis, 0, slice(None, None, None))]
    getitem_111 = view_25[(Ellipsis, 1, slice(None, None, None))]
    getitem_112 = view_25[(Ellipsis, 2, slice(None, None, None))];
    view_25 = None
    size_32 = getitem_110.size()
    getitem_113 = size_32[0]
    getitem_114 = size_32[1]
    getitem_115 = size_32[2]
    getitem_116 = size_32[3];
    size_32 = None
    transpose_12 = getitem_110.transpose(1, 2);
    getitem_110 = None
    mul_63 = getitem_113 * 16
    reshape_25 = transpose_12.reshape(mul_63, getitem_114, 64);
    transpose_12 = mul_63 = None
    permute_12 = getitem_111.permute(0, 2, 3, 1);
    getitem_111 = None
    mul_64 = getitem_113 * 16
    reshape_26 = permute_12.reshape(mul_64, 64, getitem_114);
    permute_12 = mul_64 = None
    transpose_13 = getitem_112.transpose(1, 2);
    getitem_112 = None
    mul_65 = getitem_113 * 16
    reshape_27 = transpose_13.reshape(mul_65, getitem_114, 64);
    transpose_13 = mul_65 = None
    size_33 = reshape_26.size()
    getitem_117 = size_33[0]
    getitem_118 = size_33[1]
    getitem_119 = size_33[2];
    size_33 = None
    baddbmm_6 = to.baddbmm(batch1=reshape_25, batch2=reshape_26, beta=1.0, alpha=0.125);
    reshape_25 = reshape_26 = None
    view_26 = baddbmm_6.view(getitem_113, 16, getitem_114, getitem_119);
    baddbmm_6 = None
    getattr_29 = view_26.dtype
    eq_6 = getattr_29 == torch.float16
    getattr_30 = view_26.dtype
    finfo_9 = torch.finfo(getattr_30);
    getattr_30 = None
    getattr_31 = finfo_9.min;
    finfo_9 = None
    masked_fill_8 = torch.masked_fill(view_26, bool_2, getattr_31);
    view_26 = getattr_31 = None
    softmax_6 = torch.nn.functional.softmax(masked_fill_8, dim=-1, _stacklevel=3, dtype=torch.float32);
    masked_fill_8 = None
    to_11 = softmax_6.to(getattr_29);
    softmax_6 = getattr_29 = None
    transformer_h_6_self_attention_attention_dropout = getattr(self.transformer.h,
                                                               "6").self_attention.attention_dropout(to_11);
    to_11 = None
    mul_66 = getitem_113 * 16;
    getitem_113 = None
    view_27 = transformer_h_6_self_attention_attention_dropout.view(mul_66, getitem_114, getitem_119);
    transformer_h_6_self_attention_attention_dropout = mul_66 = getitem_114 = getitem_119 = None
    bmm_6 = torch.bmm(view_27, reshape_27);
    view_27 = reshape_27 = None
    size_34 = bmm_6.size()
    getitem_120 = size_34[0]
    getitem_121 = size_34[1]
    getitem_122 = size_34[2];
    size_34 = None
    floordiv_6 = getitem_120 // 16;
    getitem_120 = None
    view_28 = bmm_6.view(floordiv_6, 16, getitem_121, 64);
    bmm_6 = None
    permute_13 = view_28.permute(0, 2, 1, 3);
    view_28 = None
    reshape_28 = permute_13.reshape(floordiv_6, getitem_121, 1024);
    permute_13 = floordiv_6 = getitem_121 = None
    transformer_h_6_self_attention_dense = getattr(self.transformer.h, "6").self_attention.dense(reshape_28);
    reshape_28 = None
    dropout_12 = torch.nn.functional.dropout(transformer_h_6_self_attention_dense, p=0.0, training=False,
                                             inplace=False);
    transformer_h_6_self_attention_dense = None
    add_27 = add_26 + dropout_12;
    add_26 = dropout_12 = None
    transformer_h_6_post_attention_layernorm = getattr(self.transformer.h, "6").post_attention_layernorm(add_27)
    transformer_h_6_mlp_dense_h_to_4h = getattr(self.transformer.h, "6").mlp.dense_h_to_4h(
        transformer_h_6_post_attention_layernorm);
    transformer_h_6_post_attention_layernorm = None
    mul_67 = transformer_h_6_mlp_dense_h_to_4h * 0.5
    mul_68 = 0.79788456 * transformer_h_6_mlp_dense_h_to_4h
    mul_69 = 0.044715 * transformer_h_6_mlp_dense_h_to_4h
    mul_70 = mul_69 * transformer_h_6_mlp_dense_h_to_4h;
    mul_69 = transformer_h_6_mlp_dense_h_to_4h = None
    add_28 = 1 + mul_70;
    mul_70 = None
    mul_71 = mul_68 * add_28;
    mul_68 = add_28 = None
    tanh_6 = torch.tanh(mul_71);
    mul_71 = None
    add_29 = 1.0 + tanh_6;
    tanh_6 = None
    mul_72 = mul_67 * add_29;
    mul_67 = add_29 = None
    transformer_h_6_mlp_dense_4h_to_h = getattr(self.transformer.h, "6").mlp.dense_4h_to_h(mul_72);
    mul_72 = None
    dropout_13 = torch.nn.functional.dropout(transformer_h_6_mlp_dense_4h_to_h, p=0.0, training=False, inplace=False);
    transformer_h_6_mlp_dense_4h_to_h = None
    add_30 = add_27 + dropout_13;
    add_27 = dropout_13 = None
    transformer_h_7_input_layernorm = getattr(self.transformer.h, "7").input_layernorm(add_30)
    transformer_h_7_self_attention_query_key_value = getattr(self.transformer.h, "7").self_attention.query_key_value(
        transformer_h_7_input_layernorm);
    transformer_h_7_input_layernorm = None
    size_35 = transformer_h_7_self_attention_query_key_value.size()
    getitem_123 = size_35[0]
    getitem_124 = size_35[1]
    getitem_125 = size_35[2];
    size_35 = None
    view_29 = transformer_h_7_self_attention_query_key_value.view(getitem_123, getitem_124, 16, 3, 64);
    transformer_h_7_self_attention_query_key_value = getitem_123 = getitem_124 = None
    getitem_126 = view_29[(Ellipsis, 0, slice(None, None, None))]
    getitem_127 = view_29[(Ellipsis, 1, slice(None, None, None))]
    getitem_128 = view_29[(Ellipsis, 2, slice(None, None, None))];
    view_29 = None
    size_36 = getitem_126.size()
    getitem_129 = size_36[0]
    getitem_130 = size_36[1]
    getitem_131 = size_36[2]
    getitem_132 = size_36[3];
    size_36 = None
    transpose_14 = getitem_126.transpose(1, 2);
    getitem_126 = None
    mul_73 = getitem_129 * 16
    reshape_29 = transpose_14.reshape(mul_73, getitem_130, 64);
    transpose_14 = mul_73 = None
    permute_14 = getitem_127.permute(0, 2, 3, 1);
    getitem_127 = None
    mul_74 = getitem_129 * 16
    reshape_30 = permute_14.reshape(mul_74, 64, getitem_130);
    permute_14 = mul_74 = None
    transpose_15 = getitem_128.transpose(1, 2);
    getitem_128 = None
    mul_75 = getitem_129 * 16
    reshape_31 = transpose_15.reshape(mul_75, getitem_130, 64);
    transpose_15 = mul_75 = None
    size_37 = reshape_30.size()
    getitem_133 = size_37[0]
    getitem_134 = size_37[1]
    getitem_135 = size_37[2];
    size_37 = None
    baddbmm_7 = to.baddbmm(batch1=reshape_29, batch2=reshape_30, beta=1.0, alpha=0.125);
    reshape_29 = reshape_30 = None
    view_30 = baddbmm_7.view(getitem_129, 16, getitem_130, getitem_135);
    baddbmm_7 = None
    getattr_32 = view_30.dtype
    eq_7 = getattr_32 == torch.float16
    getattr_33 = view_30.dtype
    finfo_10 = torch.finfo(getattr_33);
    getattr_33 = None
    getattr_34 = finfo_10.min;
    finfo_10 = None
    masked_fill_9 = torch.masked_fill(view_30, bool_2, getattr_34);
    view_30 = getattr_34 = None
    softmax_7 = torch.nn.functional.softmax(masked_fill_9, dim=-1, _stacklevel=3, dtype=torch.float32);
    masked_fill_9 = None
    to_12 = softmax_7.to(getattr_32);
    softmax_7 = getattr_32 = None
    transformer_h_7_self_attention_attention_dropout = getattr(self.transformer.h,
                                                               "7").self_attention.attention_dropout(to_12);
    to_12 = None
    mul_76 = getitem_129 * 16;
    getitem_129 = None
    view_31 = transformer_h_7_self_attention_attention_dropout.view(mul_76, getitem_130, getitem_135);
    transformer_h_7_self_attention_attention_dropout = mul_76 = getitem_130 = getitem_135 = None
    bmm_7 = torch.bmm(view_31, reshape_31);
    view_31 = reshape_31 = None
    size_38 = bmm_7.size()
    getitem_136 = size_38[0]
    getitem_137 = size_38[1]
    getitem_138 = size_38[2];
    size_38 = None
    floordiv_7 = getitem_136 // 16;
    getitem_136 = None
    view_32 = bmm_7.view(floordiv_7, 16, getitem_137, 64);
    bmm_7 = None
    permute_15 = view_32.permute(0, 2, 1, 3);
    view_32 = None
    reshape_32 = permute_15.reshape(floordiv_7, getitem_137, 1024);
    permute_15 = floordiv_7 = getitem_137 = None
    transformer_h_7_self_attention_dense = getattr(self.transformer.h, "7").self_attention.dense(reshape_32);
    reshape_32 = None
    dropout_14 = torch.nn.functional.dropout(transformer_h_7_self_attention_dense, p=0.0, training=False,
                                             inplace=False);
    transformer_h_7_self_attention_dense = None
    add_31 = add_30 + dropout_14;
    add_30 = dropout_14 = None
    transformer_h_7_post_attention_layernorm = getattr(self.transformer.h, "7").post_attention_layernorm(add_31)
    transformer_h_7_mlp_dense_h_to_4h = getattr(self.transformer.h, "7").mlp.dense_h_to_4h(
        transformer_h_7_post_attention_layernorm);
    transformer_h_7_post_attention_layernorm = None
    mul_77 = transformer_h_7_mlp_dense_h_to_4h * 0.5
    mul_78 = 0.79788456 * transformer_h_7_mlp_dense_h_to_4h
    mul_79 = 0.044715 * transformer_h_7_mlp_dense_h_to_4h
    mul_80 = mul_79 * transformer_h_7_mlp_dense_h_to_4h;
    mul_79 = transformer_h_7_mlp_dense_h_to_4h = None
    add_32 = 1 + mul_80;
    mul_80 = None
    mul_81 = mul_78 * add_32;
    mul_78 = add_32 = None
    tanh_7 = torch.tanh(mul_81);
    mul_81 = None
    add_33 = 1.0 + tanh_7;
    tanh_7 = None
    mul_82 = mul_77 * add_33;
    mul_77 = add_33 = None
    transformer_h_7_mlp_dense_4h_to_h = getattr(self.transformer.h, "7").mlp.dense_4h_to_h(mul_82);
    mul_82 = None
    dropout_15 = torch.nn.functional.dropout(transformer_h_7_mlp_dense_4h_to_h, p=0.0, training=False, inplace=False);
    transformer_h_7_mlp_dense_4h_to_h = None
    add_34 = add_31 + dropout_15;
    add_31 = dropout_15 = None
    transformer_h_8_input_layernorm = getattr(self.transformer.h, "8").input_layernorm(add_34)
    transformer_h_8_self_attention_query_key_value = getattr(self.transformer.h, "8").self_attention.query_key_value(
        transformer_h_8_input_layernorm);
    transformer_h_8_input_layernorm = None
    size_39 = transformer_h_8_self_attention_query_key_value.size()
    getitem_139 = size_39[0]
    getitem_140 = size_39[1]
    getitem_141 = size_39[2];
    size_39 = None
    view_33 = transformer_h_8_self_attention_query_key_value.view(getitem_139, getitem_140, 16, 3, 64);
    transformer_h_8_self_attention_query_key_value = getitem_139 = getitem_140 = None
    getitem_142 = view_33[(Ellipsis, 0, slice(None, None, None))]
    getitem_143 = view_33[(Ellipsis, 1, slice(None, None, None))]
    getitem_144 = view_33[(Ellipsis, 2, slice(None, None, None))];
    view_33 = None
    size_40 = getitem_142.size()
    getitem_145 = size_40[0]
    getitem_146 = size_40[1]
    getitem_147 = size_40[2]
    getitem_148 = size_40[3];
    size_40 = None
    transpose_16 = getitem_142.transpose(1, 2);
    getitem_142 = None
    mul_83 = getitem_145 * 16
    reshape_33 = transpose_16.reshape(mul_83, getitem_146, 64);
    transpose_16 = mul_83 = None
    permute_16 = getitem_143.permute(0, 2, 3, 1);
    getitem_143 = None
    mul_84 = getitem_145 * 16
    reshape_34 = permute_16.reshape(mul_84, 64, getitem_146);
    permute_16 = mul_84 = None
    transpose_17 = getitem_144.transpose(1, 2);
    getitem_144 = None
    mul_85 = getitem_145 * 16
    reshape_35 = transpose_17.reshape(mul_85, getitem_146, 64);
    transpose_17 = mul_85 = None
    size_41 = reshape_34.size()
    getitem_149 = size_41[0]
    getitem_150 = size_41[1]
    getitem_151 = size_41[2];
    size_41 = None
    baddbmm_8 = to.baddbmm(batch1=reshape_33, batch2=reshape_34, beta=1.0, alpha=0.125);
    reshape_33 = reshape_34 = None
    view_34 = baddbmm_8.view(getitem_145, 16, getitem_146, getitem_151);
    baddbmm_8 = None
    getattr_35 = view_34.dtype
    eq_8 = getattr_35 == torch.float16
    getattr_36 = view_34.dtype
    finfo_11 = torch.finfo(getattr_36);
    getattr_36 = None
    getattr_37 = finfo_11.min;
    finfo_11 = None
    masked_fill_10 = torch.masked_fill(view_34, bool_2, getattr_37);
    view_34 = getattr_37 = None
    softmax_8 = torch.nn.functional.softmax(masked_fill_10, dim=-1, _stacklevel=3, dtype=torch.float32);
    masked_fill_10 = None
    to_13 = softmax_8.to(getattr_35);
    softmax_8 = getattr_35 = None
    transformer_h_8_self_attention_attention_dropout = getattr(self.transformer.h,
                                                               "8").self_attention.attention_dropout(to_13);
    to_13 = None
    mul_86 = getitem_145 * 16;
    getitem_145 = None
    view_35 = transformer_h_8_self_attention_attention_dropout.view(mul_86, getitem_146, getitem_151);
    transformer_h_8_self_attention_attention_dropout = mul_86 = getitem_146 = getitem_151 = None
    bmm_8 = torch.bmm(view_35, reshape_35);
    view_35 = reshape_35 = None
    size_42 = bmm_8.size()
    getitem_152 = size_42[0]
    getitem_153 = size_42[1]
    getitem_154 = size_42[2];
    size_42 = None
    floordiv_8 = getitem_152 // 16;
    getitem_152 = None
    view_36 = bmm_8.view(floordiv_8, 16, getitem_153, 64);
    bmm_8 = None
    permute_17 = view_36.permute(0, 2, 1, 3);
    view_36 = None
    reshape_36 = permute_17.reshape(floordiv_8, getitem_153, 1024);
    permute_17 = floordiv_8 = getitem_153 = None
    transformer_h_8_self_attention_dense = getattr(self.transformer.h, "8").self_attention.dense(reshape_36);
    reshape_36 = None
    dropout_16 = torch.nn.functional.dropout(transformer_h_8_self_attention_dense, p=0.0, training=False,
                                             inplace=False);
    transformer_h_8_self_attention_dense = None
    add_35 = add_34 + dropout_16;
    add_34 = dropout_16 = None
    transformer_h_8_post_attention_layernorm = getattr(self.transformer.h, "8").post_attention_layernorm(add_35)
    transformer_h_8_mlp_dense_h_to_4h = getattr(self.transformer.h, "8").mlp.dense_h_to_4h(
        transformer_h_8_post_attention_layernorm);
    transformer_h_8_post_attention_layernorm = None
    mul_87 = transformer_h_8_mlp_dense_h_to_4h * 0.5
    mul_88 = 0.79788456 * transformer_h_8_mlp_dense_h_to_4h
    mul_89 = 0.044715 * transformer_h_8_mlp_dense_h_to_4h
    mul_90 = mul_89 * transformer_h_8_mlp_dense_h_to_4h;
    mul_89 = transformer_h_8_mlp_dense_h_to_4h = None
    add_36 = 1 + mul_90;
    mul_90 = None
    mul_91 = mul_88 * add_36;
    mul_88 = add_36 = None
    tanh_8 = torch.tanh(mul_91);
    mul_91 = None
    add_37 = 1.0 + tanh_8;
    tanh_8 = None
    mul_92 = mul_87 * add_37;
    mul_87 = add_37 = None
    transformer_h_8_mlp_dense_4h_to_h = getattr(self.transformer.h, "8").mlp.dense_4h_to_h(mul_92);
    mul_92 = None
    dropout_17 = torch.nn.functional.dropout(transformer_h_8_mlp_dense_4h_to_h, p=0.0, training=False, inplace=False);
    transformer_h_8_mlp_dense_4h_to_h = None
    add_38 = add_35 + dropout_17;
    add_35 = dropout_17 = None
    transformer_h_9_input_layernorm = getattr(self.transformer.h, "9").input_layernorm(add_38)
    transformer_h_9_self_attention_query_key_value = getattr(self.transformer.h, "9").self_attention.query_key_value(
        transformer_h_9_input_layernorm);
    transformer_h_9_input_layernorm = None
    size_43 = transformer_h_9_self_attention_query_key_value.size()
    getitem_155 = size_43[0]
    getitem_156 = size_43[1]
    getitem_157 = size_43[2];
    size_43 = None
    view_37 = transformer_h_9_self_attention_query_key_value.view(getitem_155, getitem_156, 16, 3, 64);
    transformer_h_9_self_attention_query_key_value = getitem_155 = getitem_156 = None
    getitem_158 = view_37[(Ellipsis, 0, slice(None, None, None))]
    getitem_159 = view_37[(Ellipsis, 1, slice(None, None, None))]
    getitem_160 = view_37[(Ellipsis, 2, slice(None, None, None))];
    view_37 = None
    size_44 = getitem_158.size()
    getitem_161 = size_44[0]
    getitem_162 = size_44[1]
    getitem_163 = size_44[2]
    getitem_164 = size_44[3];
    size_44 = None
    transpose_18 = getitem_158.transpose(1, 2);
    getitem_158 = None
    mul_93 = getitem_161 * 16
    reshape_37 = transpose_18.reshape(mul_93, getitem_162, 64);
    transpose_18 = mul_93 = None
    permute_18 = getitem_159.permute(0, 2, 3, 1);
    getitem_159 = None
    mul_94 = getitem_161 * 16
    reshape_38 = permute_18.reshape(mul_94, 64, getitem_162);
    permute_18 = mul_94 = None
    transpose_19 = getitem_160.transpose(1, 2);
    getitem_160 = None
    mul_95 = getitem_161 * 16
    reshape_39 = transpose_19.reshape(mul_95, getitem_162, 64);
    transpose_19 = mul_95 = None
    size_45 = reshape_38.size()
    getitem_165 = size_45[0]
    getitem_166 = size_45[1]
    getitem_167 = size_45[2];
    size_45 = None
    baddbmm_9 = to.baddbmm(batch1=reshape_37, batch2=reshape_38, beta=1.0, alpha=0.125);
    reshape_37 = reshape_38 = None
    view_38 = baddbmm_9.view(getitem_161, 16, getitem_162, getitem_167);
    baddbmm_9 = None
    getattr_38 = view_38.dtype
    eq_9 = getattr_38 == torch.float16
    getattr_39 = view_38.dtype
    finfo_12 = torch.finfo(getattr_39);
    getattr_39 = None
    getattr_40 = finfo_12.min;
    finfo_12 = None
    masked_fill_11 = torch.masked_fill(view_38, bool_2, getattr_40);
    view_38 = getattr_40 = None
    softmax_9 = torch.nn.functional.softmax(masked_fill_11, dim=-1, _stacklevel=3, dtype=torch.float32);
    masked_fill_11 = None
    to_14 = softmax_9.to(getattr_38);
    softmax_9 = getattr_38 = None
    transformer_h_9_self_attention_attention_dropout = getattr(self.transformer.h,
                                                               "9").self_attention.attention_dropout(to_14);
    to_14 = None
    mul_96 = getitem_161 * 16;
    getitem_161 = None
    view_39 = transformer_h_9_self_attention_attention_dropout.view(mul_96, getitem_162, getitem_167);
    transformer_h_9_self_attention_attention_dropout = mul_96 = getitem_162 = getitem_167 = None
    bmm_9 = torch.bmm(view_39, reshape_39);
    view_39 = reshape_39 = None
    size_46 = bmm_9.size()
    getitem_168 = size_46[0]
    getitem_169 = size_46[1]
    getitem_170 = size_46[2];
    size_46 = None
    floordiv_9 = getitem_168 // 16;
    getitem_168 = None
    view_40 = bmm_9.view(floordiv_9, 16, getitem_169, 64);
    bmm_9 = None
    permute_19 = view_40.permute(0, 2, 1, 3);
    view_40 = None
    reshape_40 = permute_19.reshape(floordiv_9, getitem_169, 1024);
    permute_19 = floordiv_9 = getitem_169 = None
    transformer_h_9_self_attention_dense = getattr(self.transformer.h, "9").self_attention.dense(reshape_40);
    reshape_40 = None
    dropout_18 = torch.nn.functional.dropout(transformer_h_9_self_attention_dense, p=0.0, training=False,
                                             inplace=False);
    transformer_h_9_self_attention_dense = None
    add_39 = add_38 + dropout_18;
    add_38 = dropout_18 = None
    transformer_h_9_post_attention_layernorm = getattr(self.transformer.h, "9").post_attention_layernorm(add_39)
    transformer_h_9_mlp_dense_h_to_4h = getattr(self.transformer.h, "9").mlp.dense_h_to_4h(
        transformer_h_9_post_attention_layernorm);
    transformer_h_9_post_attention_layernorm = None
    mul_97 = transformer_h_9_mlp_dense_h_to_4h * 0.5
    mul_98 = 0.79788456 * transformer_h_9_mlp_dense_h_to_4h
    mul_99 = 0.044715 * transformer_h_9_mlp_dense_h_to_4h
    mul_100 = mul_99 * transformer_h_9_mlp_dense_h_to_4h;
    mul_99 = transformer_h_9_mlp_dense_h_to_4h = None
    add_40 = 1 + mul_100;
    mul_100 = None
    mul_101 = mul_98 * add_40;
    mul_98 = add_40 = None
    tanh_9 = torch.tanh(mul_101);
    mul_101 = None
    add_41 = 1.0 + tanh_9;
    tanh_9 = None
    mul_102 = mul_97 * add_41;
    mul_97 = add_41 = None
    transformer_h_9_mlp_dense_4h_to_h = getattr(self.transformer.h, "9").mlp.dense_4h_to_h(mul_102);
    mul_102 = None
    dropout_19 = torch.nn.functional.dropout(transformer_h_9_mlp_dense_4h_to_h, p=0.0, training=False, inplace=False);
    transformer_h_9_mlp_dense_4h_to_h = None
    add_42 = add_39 + dropout_19;
    add_39 = dropout_19 = None
    transformer_h_10_input_layernorm = getattr(self.transformer.h, "10").input_layernorm(add_42)
    transformer_h_10_self_attention_query_key_value = getattr(self.transformer.h, "10").self_attention.query_key_value(
        transformer_h_10_input_layernorm);
    transformer_h_10_input_layernorm = None
    size_47 = transformer_h_10_self_attention_query_key_value.size()
    getitem_171 = size_47[0]
    getitem_172 = size_47[1]
    getitem_173 = size_47[2];
    size_47 = None
    view_41 = transformer_h_10_self_attention_query_key_value.view(getitem_171, getitem_172, 16, 3, 64);
    transformer_h_10_self_attention_query_key_value = getitem_171 = getitem_172 = None
    getitem_174 = view_41[(Ellipsis, 0, slice(None, None, None))]
    getitem_175 = view_41[(Ellipsis, 1, slice(None, None, None))]
    getitem_176 = view_41[(Ellipsis, 2, slice(None, None, None))];
    view_41 = None
    size_48 = getitem_174.size()
    getitem_177 = size_48[0]
    getitem_178 = size_48[1]
    getitem_179 = size_48[2]
    getitem_180 = size_48[3];
    size_48 = None
    transpose_20 = getitem_174.transpose(1, 2);
    getitem_174 = None
    mul_103 = getitem_177 * 16
    reshape_41 = transpose_20.reshape(mul_103, getitem_178, 64);
    transpose_20 = mul_103 = None
    permute_20 = getitem_175.permute(0, 2, 3, 1);
    getitem_175 = None
    mul_104 = getitem_177 * 16
    reshape_42 = permute_20.reshape(mul_104, 64, getitem_178);
    permute_20 = mul_104 = None
    transpose_21 = getitem_176.transpose(1, 2);
    getitem_176 = None
    mul_105 = getitem_177 * 16
    reshape_43 = transpose_21.reshape(mul_105, getitem_178, 64);
    transpose_21 = mul_105 = None
    size_49 = reshape_42.size()
    getitem_181 = size_49[0]
    getitem_182 = size_49[1]
    getitem_183 = size_49[2];
    size_49 = None
    baddbmm_10 = to.baddbmm(batch1=reshape_41, batch2=reshape_42, beta=1.0, alpha=0.125);
    reshape_41 = reshape_42 = None
    view_42 = baddbmm_10.view(getitem_177, 16, getitem_178, getitem_183);
    baddbmm_10 = None
    getattr_41 = view_42.dtype
    eq_10 = getattr_41 == torch.float16
    getattr_42 = view_42.dtype
    finfo_13 = torch.finfo(getattr_42);
    getattr_42 = None
    getattr_43 = finfo_13.min;
    finfo_13 = None
    masked_fill_12 = torch.masked_fill(view_42, bool_2, getattr_43);
    view_42 = getattr_43 = None
    softmax_10 = torch.nn.functional.softmax(masked_fill_12, dim=-1, _stacklevel=3, dtype=torch.float32);
    masked_fill_12 = None
    to_15 = softmax_10.to(getattr_41);
    softmax_10 = getattr_41 = None
    transformer_h_10_self_attention_attention_dropout = getattr(self.transformer.h,
                                                                "10").self_attention.attention_dropout(to_15);
    to_15 = None
    mul_106 = getitem_177 * 16;
    getitem_177 = None
    view_43 = transformer_h_10_self_attention_attention_dropout.view(mul_106, getitem_178, getitem_183);
    transformer_h_10_self_attention_attention_dropout = mul_106 = getitem_178 = getitem_183 = None
    bmm_10 = torch.bmm(view_43, reshape_43);
    view_43 = reshape_43 = None
    size_50 = bmm_10.size()
    getitem_184 = size_50[0]
    getitem_185 = size_50[1]
    getitem_186 = size_50[2];
    size_50 = None
    floordiv_10 = getitem_184 // 16;
    getitem_184 = None
    view_44 = bmm_10.view(floordiv_10, 16, getitem_185, 64);
    bmm_10 = None
    permute_21 = view_44.permute(0, 2, 1, 3);
    view_44 = None
    reshape_44 = permute_21.reshape(floordiv_10, getitem_185, 1024);
    permute_21 = floordiv_10 = getitem_185 = None
    transformer_h_10_self_attention_dense = getattr(self.transformer.h, "10").self_attention.dense(reshape_44);
    reshape_44 = None
    dropout_20 = torch.nn.functional.dropout(transformer_h_10_self_attention_dense, p=0.0, training=False,
                                             inplace=False);
    transformer_h_10_self_attention_dense = None
    add_43 = add_42 + dropout_20;
    add_42 = dropout_20 = None
    transformer_h_10_post_attention_layernorm = getattr(self.transformer.h, "10").post_attention_layernorm(add_43)
    transformer_h_10_mlp_dense_h_to_4h = getattr(self.transformer.h, "10").mlp.dense_h_to_4h(
        transformer_h_10_post_attention_layernorm);
    transformer_h_10_post_attention_layernorm = None
    mul_107 = transformer_h_10_mlp_dense_h_to_4h * 0.5
    mul_108 = 0.79788456 * transformer_h_10_mlp_dense_h_to_4h
    mul_109 = 0.044715 * transformer_h_10_mlp_dense_h_to_4h
    mul_110 = mul_109 * transformer_h_10_mlp_dense_h_to_4h;
    mul_109 = transformer_h_10_mlp_dense_h_to_4h = None
    add_44 = 1 + mul_110;
    mul_110 = None
    mul_111 = mul_108 * add_44;
    mul_108 = add_44 = None
    tanh_10 = torch.tanh(mul_111);
    mul_111 = None
    add_45 = 1.0 + tanh_10;
    tanh_10 = None
    mul_112 = mul_107 * add_45;
    mul_107 = add_45 = None
    transformer_h_10_mlp_dense_4h_to_h = getattr(self.transformer.h, "10").mlp.dense_4h_to_h(mul_112);
    mul_112 = None
    dropout_21 = torch.nn.functional.dropout(transformer_h_10_mlp_dense_4h_to_h, p=0.0, training=False, inplace=False);
    transformer_h_10_mlp_dense_4h_to_h = None
    add_46 = add_43 + dropout_21;
    add_43 = dropout_21 = None
    transformer_h_11_input_layernorm = getattr(self.transformer.h, "11").input_layernorm(add_46)
    transformer_h_11_self_attention_query_key_value = getattr(self.transformer.h, "11").self_attention.query_key_value(
        transformer_h_11_input_layernorm);
    transformer_h_11_input_layernorm = None
    size_51 = transformer_h_11_self_attention_query_key_value.size()
    getitem_187 = size_51[0]
    getitem_188 = size_51[1]
    getitem_189 = size_51[2];
    size_51 = None
    view_45 = transformer_h_11_self_attention_query_key_value.view(getitem_187, getitem_188, 16, 3, 64);
    transformer_h_11_self_attention_query_key_value = getitem_187 = getitem_188 = None
    getitem_190 = view_45[(Ellipsis, 0, slice(None, None, None))]
    getitem_191 = view_45[(Ellipsis, 1, slice(None, None, None))]
    getitem_192 = view_45[(Ellipsis, 2, slice(None, None, None))];
    view_45 = None
    size_52 = getitem_190.size()
    getitem_193 = size_52[0]
    getitem_194 = size_52[1]
    getitem_195 = size_52[2]
    getitem_196 = size_52[3];
    size_52 = None
    transpose_22 = getitem_190.transpose(1, 2);
    getitem_190 = None
    mul_113 = getitem_193 * 16
    reshape_45 = transpose_22.reshape(mul_113, getitem_194, 64);
    transpose_22 = mul_113 = None
    permute_22 = getitem_191.permute(0, 2, 3, 1);
    getitem_191 = None
    mul_114 = getitem_193 * 16
    reshape_46 = permute_22.reshape(mul_114, 64, getitem_194);
    permute_22 = mul_114 = None
    transpose_23 = getitem_192.transpose(1, 2);
    getitem_192 = None
    mul_115 = getitem_193 * 16
    reshape_47 = transpose_23.reshape(mul_115, getitem_194, 64);
    transpose_23 = mul_115 = None
    size_53 = reshape_46.size()
    getitem_197 = size_53[0]
    getitem_198 = size_53[1]
    getitem_199 = size_53[2];
    size_53 = None
    baddbmm_11 = to.baddbmm(batch1=reshape_45, batch2=reshape_46, beta=1.0, alpha=0.125);
    reshape_45 = reshape_46 = None
    view_46 = baddbmm_11.view(getitem_193, 16, getitem_194, getitem_199);
    baddbmm_11 = None
    getattr_44 = view_46.dtype
    eq_11 = getattr_44 == torch.float16
    getattr_45 = view_46.dtype
    finfo_14 = torch.finfo(getattr_45);
    getattr_45 = None
    getattr_46 = finfo_14.min;
    finfo_14 = None
    masked_fill_13 = torch.masked_fill(view_46, bool_2, getattr_46);
    view_46 = getattr_46 = None
    softmax_11 = torch.nn.functional.softmax(masked_fill_13, dim=-1, _stacklevel=3, dtype=torch.float32);
    masked_fill_13 = None
    to_16 = softmax_11.to(getattr_44);
    softmax_11 = getattr_44 = None
    transformer_h_11_self_attention_attention_dropout = getattr(self.transformer.h,
                                                                "11").self_attention.attention_dropout(to_16);
    to_16 = None
    mul_116 = getitem_193 * 16;
    getitem_193 = None
    view_47 = transformer_h_11_self_attention_attention_dropout.view(mul_116, getitem_194, getitem_199);
    transformer_h_11_self_attention_attention_dropout = mul_116 = getitem_194 = getitem_199 = None
    bmm_11 = torch.bmm(view_47, reshape_47);
    view_47 = reshape_47 = None
    size_54 = bmm_11.size()
    getitem_200 = size_54[0]
    getitem_201 = size_54[1]
    getitem_202 = size_54[2];
    size_54 = None
    floordiv_11 = getitem_200 // 16;
    getitem_200 = None
    view_48 = bmm_11.view(floordiv_11, 16, getitem_201, 64);
    bmm_11 = None
    permute_23 = view_48.permute(0, 2, 1, 3);
    view_48 = None
    reshape_48 = permute_23.reshape(floordiv_11, getitem_201, 1024);
    permute_23 = floordiv_11 = getitem_201 = None
    transformer_h_11_self_attention_dense = getattr(self.transformer.h, "11").self_attention.dense(reshape_48);
    reshape_48 = None
    dropout_22 = torch.nn.functional.dropout(transformer_h_11_self_attention_dense, p=0.0, training=False,
                                             inplace=False);
    transformer_h_11_self_attention_dense = None
    add_47 = add_46 + dropout_22;
    add_46 = dropout_22 = None
    transformer_h_11_post_attention_layernorm = getattr(self.transformer.h, "11").post_attention_layernorm(add_47)
    transformer_h_11_mlp_dense_h_to_4h = getattr(self.transformer.h, "11").mlp.dense_h_to_4h(
        transformer_h_11_post_attention_layernorm);
    transformer_h_11_post_attention_layernorm = None
    mul_117 = transformer_h_11_mlp_dense_h_to_4h * 0.5
    mul_118 = 0.79788456 * transformer_h_11_mlp_dense_h_to_4h
    mul_119 = 0.044715 * transformer_h_11_mlp_dense_h_to_4h
    mul_120 = mul_119 * transformer_h_11_mlp_dense_h_to_4h;
    mul_119 = transformer_h_11_mlp_dense_h_to_4h = None
    add_48 = 1 + mul_120;
    mul_120 = None
    mul_121 = mul_118 * add_48;
    mul_118 = add_48 = None
    tanh_11 = torch.tanh(mul_121);
    mul_121 = None
    add_49 = 1.0 + tanh_11;
    tanh_11 = None
    mul_122 = mul_117 * add_49;
    mul_117 = add_49 = None
    transformer_h_11_mlp_dense_4h_to_h = getattr(self.transformer.h, "11").mlp.dense_4h_to_h(mul_122);
    mul_122 = None
    dropout_23 = torch.nn.functional.dropout(transformer_h_11_mlp_dense_4h_to_h, p=0.0, training=False, inplace=False);
    transformer_h_11_mlp_dense_4h_to_h = None
    add_50 = add_47 + dropout_23;
    add_47 = dropout_23 = None
    transformer_h_12_input_layernorm = getattr(self.transformer.h, "12").input_layernorm(add_50)
    transformer_h_12_self_attention_query_key_value = getattr(self.transformer.h, "12").self_attention.query_key_value(
        transformer_h_12_input_layernorm);
    transformer_h_12_input_layernorm = None
    size_55 = transformer_h_12_self_attention_query_key_value.size()
    getitem_203 = size_55[0]
    getitem_204 = size_55[1]
    getitem_205 = size_55[2];
    size_55 = None
    view_49 = transformer_h_12_self_attention_query_key_value.view(getitem_203, getitem_204, 16, 3, 64);
    transformer_h_12_self_attention_query_key_value = getitem_203 = getitem_204 = None
    getitem_206 = view_49[(Ellipsis, 0, slice(None, None, None))]
    getitem_207 = view_49[(Ellipsis, 1, slice(None, None, None))]
    getitem_208 = view_49[(Ellipsis, 2, slice(None, None, None))];
    view_49 = None
    size_56 = getitem_206.size()
    getitem_209 = size_56[0]
    getitem_210 = size_56[1]
    getitem_211 = size_56[2]
    getitem_212 = size_56[3];
    size_56 = None
    transpose_24 = getitem_206.transpose(1, 2);
    getitem_206 = None
    mul_123 = getitem_209 * 16
    reshape_49 = transpose_24.reshape(mul_123, getitem_210, 64);
    transpose_24 = mul_123 = None
    permute_24 = getitem_207.permute(0, 2, 3, 1);
    getitem_207 = None
    mul_124 = getitem_209 * 16
    reshape_50 = permute_24.reshape(mul_124, 64, getitem_210);
    permute_24 = mul_124 = None
    transpose_25 = getitem_208.transpose(1, 2);
    getitem_208 = None
    mul_125 = getitem_209 * 16
    reshape_51 = transpose_25.reshape(mul_125, getitem_210, 64);
    transpose_25 = mul_125 = None
    size_57 = reshape_50.size()
    getitem_213 = size_57[0]
    getitem_214 = size_57[1]
    getitem_215 = size_57[2];
    size_57 = None
    baddbmm_12 = to.baddbmm(batch1=reshape_49, batch2=reshape_50, beta=1.0, alpha=0.125);
    reshape_49 = reshape_50 = None
    view_50 = baddbmm_12.view(getitem_209, 16, getitem_210, getitem_215);
    baddbmm_12 = None
    getattr_47 = view_50.dtype
    eq_12 = getattr_47 == torch.float16
    getattr_48 = view_50.dtype
    finfo_15 = torch.finfo(getattr_48);
    getattr_48 = None
    getattr_49 = finfo_15.min;
    finfo_15 = None
    masked_fill_14 = torch.masked_fill(view_50, bool_2, getattr_49);
    view_50 = getattr_49 = None
    softmax_12 = torch.nn.functional.softmax(masked_fill_14, dim=-1, _stacklevel=3, dtype=torch.float32);
    masked_fill_14 = None
    to_17 = softmax_12.to(getattr_47);
    softmax_12 = getattr_47 = None
    transformer_h_12_self_attention_attention_dropout = getattr(self.transformer.h,
                                                                "12").self_attention.attention_dropout(to_17);
    to_17 = None
    mul_126 = getitem_209 * 16;
    getitem_209 = None
    view_51 = transformer_h_12_self_attention_attention_dropout.view(mul_126, getitem_210, getitem_215);
    transformer_h_12_self_attention_attention_dropout = mul_126 = getitem_210 = getitem_215 = None
    bmm_12 = torch.bmm(view_51, reshape_51);
    view_51 = reshape_51 = None
    size_58 = bmm_12.size()
    getitem_216 = size_58[0]
    getitem_217 = size_58[1]
    getitem_218 = size_58[2];
    size_58 = None
    floordiv_12 = getitem_216 // 16;
    getitem_216 = None
    view_52 = bmm_12.view(floordiv_12, 16, getitem_217, 64);
    bmm_12 = None
    permute_25 = view_52.permute(0, 2, 1, 3);
    view_52 = None
    reshape_52 = permute_25.reshape(floordiv_12, getitem_217, 1024);
    permute_25 = floordiv_12 = getitem_217 = None
    transformer_h_12_self_attention_dense = getattr(self.transformer.h, "12").self_attention.dense(reshape_52);
    reshape_52 = None
    dropout_24 = torch.nn.functional.dropout(transformer_h_12_self_attention_dense, p=0.0, training=False,
                                             inplace=False);
    transformer_h_12_self_attention_dense = None
    add_51 = add_50 + dropout_24;
    add_50 = dropout_24 = None
    transformer_h_12_post_attention_layernorm = getattr(self.transformer.h, "12").post_attention_layernorm(add_51)
    transformer_h_12_mlp_dense_h_to_4h = getattr(self.transformer.h, "12").mlp.dense_h_to_4h(
        transformer_h_12_post_attention_layernorm);
    transformer_h_12_post_attention_layernorm = None
    mul_127 = transformer_h_12_mlp_dense_h_to_4h * 0.5
    mul_128 = 0.79788456 * transformer_h_12_mlp_dense_h_to_4h
    mul_129 = 0.044715 * transformer_h_12_mlp_dense_h_to_4h
    mul_130 = mul_129 * transformer_h_12_mlp_dense_h_to_4h;
    mul_129 = transformer_h_12_mlp_dense_h_to_4h = None
    add_52 = 1 + mul_130;
    mul_130 = None
    mul_131 = mul_128 * add_52;
    mul_128 = add_52 = None
    tanh_12 = torch.tanh(mul_131);
    mul_131 = None
    add_53 = 1.0 + tanh_12;
    tanh_12 = None
    mul_132 = mul_127 * add_53;
    mul_127 = add_53 = None
    transformer_h_12_mlp_dense_4h_to_h = getattr(self.transformer.h, "12").mlp.dense_4h_to_h(mul_132);
    mul_132 = None
    dropout_25 = torch.nn.functional.dropout(transformer_h_12_mlp_dense_4h_to_h, p=0.0, training=False, inplace=False);
    transformer_h_12_mlp_dense_4h_to_h = None
    add_54 = add_51 + dropout_25;
    add_51 = dropout_25 = None
    transformer_h_13_input_layernorm = getattr(self.transformer.h, "13").input_layernorm(add_54)
    transformer_h_13_self_attention_query_key_value = getattr(self.transformer.h, "13").self_attention.query_key_value(
        transformer_h_13_input_layernorm);
    transformer_h_13_input_layernorm = None
    size_59 = transformer_h_13_self_attention_query_key_value.size()
    getitem_219 = size_59[0]
    getitem_220 = size_59[1]
    getitem_221 = size_59[2];
    size_59 = None
    view_53 = transformer_h_13_self_attention_query_key_value.view(getitem_219, getitem_220, 16, 3, 64);
    transformer_h_13_self_attention_query_key_value = getitem_219 = getitem_220 = None
    getitem_222 = view_53[(Ellipsis, 0, slice(None, None, None))]
    getitem_223 = view_53[(Ellipsis, 1, slice(None, None, None))]
    getitem_224 = view_53[(Ellipsis, 2, slice(None, None, None))];
    view_53 = None
    size_60 = getitem_222.size()
    getitem_225 = size_60[0]
    getitem_226 = size_60[1]
    getitem_227 = size_60[2]
    getitem_228 = size_60[3];
    size_60 = None
    transpose_26 = getitem_222.transpose(1, 2);
    getitem_222 = None
    mul_133 = getitem_225 * 16
    reshape_53 = transpose_26.reshape(mul_133, getitem_226, 64);
    transpose_26 = mul_133 = None
    permute_26 = getitem_223.permute(0, 2, 3, 1);
    getitem_223 = None
    mul_134 = getitem_225 * 16
    reshape_54 = permute_26.reshape(mul_134, 64, getitem_226);
    permute_26 = mul_134 = None
    transpose_27 = getitem_224.transpose(1, 2);
    getitem_224 = None
    mul_135 = getitem_225 * 16
    reshape_55 = transpose_27.reshape(mul_135, getitem_226, 64);
    transpose_27 = mul_135 = None
    size_61 = reshape_54.size()
    getitem_229 = size_61[0]
    getitem_230 = size_61[1]
    getitem_231 = size_61[2];
    size_61 = None
    baddbmm_13 = to.baddbmm(batch1=reshape_53, batch2=reshape_54, beta=1.0, alpha=0.125);
    reshape_53 = reshape_54 = None
    view_54 = baddbmm_13.view(getitem_225, 16, getitem_226, getitem_231);
    baddbmm_13 = None
    getattr_50 = view_54.dtype
    eq_13 = getattr_50 == torch.float16
    getattr_51 = view_54.dtype
    finfo_16 = torch.finfo(getattr_51);
    getattr_51 = None
    getattr_52 = finfo_16.min;
    finfo_16 = None
    masked_fill_15 = torch.masked_fill(view_54, bool_2, getattr_52);
    view_54 = getattr_52 = None
    softmax_13 = torch.nn.functional.softmax(masked_fill_15, dim=-1, _stacklevel=3, dtype=torch.float32);
    masked_fill_15 = None
    to_18 = softmax_13.to(getattr_50);
    softmax_13 = getattr_50 = None
    transformer_h_13_self_attention_attention_dropout = getattr(self.transformer.h,
                                                                "13").self_attention.attention_dropout(to_18);
    to_18 = None
    mul_136 = getitem_225 * 16;
    getitem_225 = None
    view_55 = transformer_h_13_self_attention_attention_dropout.view(mul_136, getitem_226, getitem_231);
    transformer_h_13_self_attention_attention_dropout = mul_136 = getitem_226 = getitem_231 = None
    bmm_13 = torch.bmm(view_55, reshape_55);
    view_55 = reshape_55 = None
    size_62 = bmm_13.size()
    getitem_232 = size_62[0]
    getitem_233 = size_62[1]
    getitem_234 = size_62[2];
    size_62 = None
    floordiv_13 = getitem_232 // 16;
    getitem_232 = None
    view_56 = bmm_13.view(floordiv_13, 16, getitem_233, 64);
    bmm_13 = None
    permute_27 = view_56.permute(0, 2, 1, 3);
    view_56 = None
    reshape_56 = permute_27.reshape(floordiv_13, getitem_233, 1024);
    permute_27 = floordiv_13 = getitem_233 = None
    transformer_h_13_self_attention_dense = getattr(self.transformer.h, "13").self_attention.dense(reshape_56);
    reshape_56 = None
    dropout_26 = torch.nn.functional.dropout(transformer_h_13_self_attention_dense, p=0.0, training=False,
                                             inplace=False);
    transformer_h_13_self_attention_dense = None
    add_55 = add_54 + dropout_26;
    add_54 = dropout_26 = None
    transformer_h_13_post_attention_layernorm = getattr(self.transformer.h, "13").post_attention_layernorm(add_55)
    transformer_h_13_mlp_dense_h_to_4h = getattr(self.transformer.h, "13").mlp.dense_h_to_4h(
        transformer_h_13_post_attention_layernorm);
    transformer_h_13_post_attention_layernorm = None
    mul_137 = transformer_h_13_mlp_dense_h_to_4h * 0.5
    mul_138 = 0.79788456 * transformer_h_13_mlp_dense_h_to_4h
    mul_139 = 0.044715 * transformer_h_13_mlp_dense_h_to_4h
    mul_140 = mul_139 * transformer_h_13_mlp_dense_h_to_4h;
    mul_139 = transformer_h_13_mlp_dense_h_to_4h = None
    add_56 = 1 + mul_140;
    mul_140 = None
    mul_141 = mul_138 * add_56;
    mul_138 = add_56 = None
    tanh_13 = torch.tanh(mul_141);
    mul_141 = None
    add_57 = 1.0 + tanh_13;
    tanh_13 = None
    mul_142 = mul_137 * add_57;
    mul_137 = add_57 = None
    transformer_h_13_mlp_dense_4h_to_h = getattr(self.transformer.h, "13").mlp.dense_4h_to_h(mul_142);
    mul_142 = None
    dropout_27 = torch.nn.functional.dropout(transformer_h_13_mlp_dense_4h_to_h, p=0.0, training=False, inplace=False);
    transformer_h_13_mlp_dense_4h_to_h = None
    add_58 = add_55 + dropout_27;
    add_55 = dropout_27 = None
    transformer_h_14_input_layernorm = getattr(self.transformer.h, "14").input_layernorm(add_58)
    transformer_h_14_self_attention_query_key_value = getattr(self.transformer.h, "14").self_attention.query_key_value(
        transformer_h_14_input_layernorm);
    transformer_h_14_input_layernorm = None
    size_63 = transformer_h_14_self_attention_query_key_value.size()
    getitem_235 = size_63[0]
    getitem_236 = size_63[1]
    getitem_237 = size_63[2];
    size_63 = None
    view_57 = transformer_h_14_self_attention_query_key_value.view(getitem_235, getitem_236, 16, 3, 64);
    transformer_h_14_self_attention_query_key_value = getitem_235 = getitem_236 = None
    getitem_238 = view_57[(Ellipsis, 0, slice(None, None, None))]
    getitem_239 = view_57[(Ellipsis, 1, slice(None, None, None))]
    getitem_240 = view_57[(Ellipsis, 2, slice(None, None, None))];
    view_57 = None
    size_64 = getitem_238.size()
    getitem_241 = size_64[0]
    getitem_242 = size_64[1]
    getitem_243 = size_64[2]
    getitem_244 = size_64[3];
    size_64 = None
    transpose_28 = getitem_238.transpose(1, 2);
    getitem_238 = None
    mul_143 = getitem_241 * 16
    reshape_57 = transpose_28.reshape(mul_143, getitem_242, 64);
    transpose_28 = mul_143 = None
    permute_28 = getitem_239.permute(0, 2, 3, 1);
    getitem_239 = None
    mul_144 = getitem_241 * 16
    reshape_58 = permute_28.reshape(mul_144, 64, getitem_242);
    permute_28 = mul_144 = None
    transpose_29 = getitem_240.transpose(1, 2);
    getitem_240 = None
    mul_145 = getitem_241 * 16
    reshape_59 = transpose_29.reshape(mul_145, getitem_242, 64);
    transpose_29 = mul_145 = None
    size_65 = reshape_58.size()
    getitem_245 = size_65[0]
    getitem_246 = size_65[1]
    getitem_247 = size_65[2];
    size_65 = None
    baddbmm_14 = to.baddbmm(batch1=reshape_57, batch2=reshape_58, beta=1.0, alpha=0.125);
    reshape_57 = reshape_58 = None
    view_58 = baddbmm_14.view(getitem_241, 16, getitem_242, getitem_247);
    baddbmm_14 = None
    getattr_53 = view_58.dtype
    eq_14 = getattr_53 == torch.float16
    getattr_54 = view_58.dtype
    finfo_17 = torch.finfo(getattr_54);
    getattr_54 = None
    getattr_55 = finfo_17.min;
    finfo_17 = None
    masked_fill_16 = torch.masked_fill(view_58, bool_2, getattr_55);
    view_58 = getattr_55 = None
    softmax_14 = torch.nn.functional.softmax(masked_fill_16, dim=-1, _stacklevel=3, dtype=torch.float32);
    masked_fill_16 = None
    to_19 = softmax_14.to(getattr_53);
    softmax_14 = getattr_53 = None
    transformer_h_14_self_attention_attention_dropout = getattr(self.transformer.h,
                                                                "14").self_attention.attention_dropout(to_19);
    to_19 = None
    mul_146 = getitem_241 * 16;
    getitem_241 = None
    view_59 = transformer_h_14_self_attention_attention_dropout.view(mul_146, getitem_242, getitem_247);
    transformer_h_14_self_attention_attention_dropout = mul_146 = getitem_242 = getitem_247 = None
    bmm_14 = torch.bmm(view_59, reshape_59);
    view_59 = reshape_59 = None
    size_66 = bmm_14.size()
    getitem_248 = size_66[0]
    getitem_249 = size_66[1]
    getitem_250 = size_66[2];
    size_66 = None
    floordiv_14 = getitem_248 // 16;
    getitem_248 = None
    view_60 = bmm_14.view(floordiv_14, 16, getitem_249, 64);
    bmm_14 = None
    permute_29 = view_60.permute(0, 2, 1, 3);
    view_60 = None
    reshape_60 = permute_29.reshape(floordiv_14, getitem_249, 1024);
    permute_29 = floordiv_14 = getitem_249 = None
    transformer_h_14_self_attention_dense = getattr(self.transformer.h, "14").self_attention.dense(reshape_60);
    reshape_60 = None
    dropout_28 = torch.nn.functional.dropout(transformer_h_14_self_attention_dense, p=0.0, training=False,
                                             inplace=False);
    transformer_h_14_self_attention_dense = None
    add_59 = add_58 + dropout_28;
    add_58 = dropout_28 = None
    transformer_h_14_post_attention_layernorm = getattr(self.transformer.h, "14").post_attention_layernorm(add_59)
    transformer_h_14_mlp_dense_h_to_4h = getattr(self.transformer.h, "14").mlp.dense_h_to_4h(
        transformer_h_14_post_attention_layernorm);
    transformer_h_14_post_attention_layernorm = None
    mul_147 = transformer_h_14_mlp_dense_h_to_4h * 0.5
    mul_148 = 0.79788456 * transformer_h_14_mlp_dense_h_to_4h
    mul_149 = 0.044715 * transformer_h_14_mlp_dense_h_to_4h
    mul_150 = mul_149 * transformer_h_14_mlp_dense_h_to_4h;
    mul_149 = transformer_h_14_mlp_dense_h_to_4h = None
    add_60 = 1 + mul_150;
    mul_150 = None
    mul_151 = mul_148 * add_60;
    mul_148 = add_60 = None
    tanh_14 = torch.tanh(mul_151);
    mul_151 = None
    add_61 = 1.0 + tanh_14;
    tanh_14 = None
    mul_152 = mul_147 * add_61;
    mul_147 = add_61 = None
    transformer_h_14_mlp_dense_4h_to_h = getattr(self.transformer.h, "14").mlp.dense_4h_to_h(mul_152);
    mul_152 = None
    dropout_29 = torch.nn.functional.dropout(transformer_h_14_mlp_dense_4h_to_h, p=0.0, training=False, inplace=False);
    transformer_h_14_mlp_dense_4h_to_h = None
    add_62 = add_59 + dropout_29;
    add_59 = dropout_29 = None
    transformer_h_15_input_layernorm = getattr(self.transformer.h, "15").input_layernorm(add_62)
    transformer_h_15_self_attention_query_key_value = getattr(self.transformer.h, "15").self_attention.query_key_value(
        transformer_h_15_input_layernorm);
    transformer_h_15_input_layernorm = None
    size_67 = transformer_h_15_self_attention_query_key_value.size()
    getitem_251 = size_67[0]
    getitem_252 = size_67[1]
    getitem_253 = size_67[2];
    size_67 = None
    view_61 = transformer_h_15_self_attention_query_key_value.view(getitem_251, getitem_252, 16, 3, 64);
    transformer_h_15_self_attention_query_key_value = getitem_251 = getitem_252 = None
    getitem_254 = view_61[(Ellipsis, 0, slice(None, None, None))]
    getitem_255 = view_61[(Ellipsis, 1, slice(None, None, None))]
    getitem_256 = view_61[(Ellipsis, 2, slice(None, None, None))];
    view_61 = None
    size_68 = getitem_254.size()
    getitem_257 = size_68[0]
    getitem_258 = size_68[1]
    getitem_259 = size_68[2]
    getitem_260 = size_68[3];
    size_68 = None
    transpose_30 = getitem_254.transpose(1, 2);
    getitem_254 = None
    mul_153 = getitem_257 * 16
    reshape_61 = transpose_30.reshape(mul_153, getitem_258, 64);
    transpose_30 = mul_153 = None
    permute_30 = getitem_255.permute(0, 2, 3, 1);
    getitem_255 = None
    mul_154 = getitem_257 * 16
    reshape_62 = permute_30.reshape(mul_154, 64, getitem_258);
    permute_30 = mul_154 = None
    transpose_31 = getitem_256.transpose(1, 2);
    getitem_256 = None
    mul_155 = getitem_257 * 16
    reshape_63 = transpose_31.reshape(mul_155, getitem_258, 64);
    transpose_31 = mul_155 = None
    size_69 = reshape_62.size()
    getitem_261 = size_69[0]
    getitem_262 = size_69[1]
    getitem_263 = size_69[2];
    size_69 = None
    baddbmm_15 = to.baddbmm(batch1=reshape_61, batch2=reshape_62, beta=1.0, alpha=0.125);
    reshape_61 = reshape_62 = None
    view_62 = baddbmm_15.view(getitem_257, 16, getitem_258, getitem_263);
    baddbmm_15 = None
    getattr_56 = view_62.dtype
    eq_15 = getattr_56 == torch.float16
    getattr_57 = view_62.dtype
    finfo_18 = torch.finfo(getattr_57);
    getattr_57 = None
    getattr_58 = finfo_18.min;
    finfo_18 = None
    masked_fill_17 = torch.masked_fill(view_62, bool_2, getattr_58);
    view_62 = getattr_58 = None
    softmax_15 = torch.nn.functional.softmax(masked_fill_17, dim=-1, _stacklevel=3, dtype=torch.float32);
    masked_fill_17 = None
    to_20 = softmax_15.to(getattr_56);
    softmax_15 = getattr_56 = None
    transformer_h_15_self_attention_attention_dropout = getattr(self.transformer.h,
                                                                "15").self_attention.attention_dropout(to_20);
    to_20 = None
    mul_156 = getitem_257 * 16;
    getitem_257 = None
    view_63 = transformer_h_15_self_attention_attention_dropout.view(mul_156, getitem_258, getitem_263);
    transformer_h_15_self_attention_attention_dropout = mul_156 = getitem_258 = getitem_263 = None
    bmm_15 = torch.bmm(view_63, reshape_63);
    view_63 = reshape_63 = None
    size_70 = bmm_15.size()
    getitem_264 = size_70[0]
    getitem_265 = size_70[1]
    getitem_266 = size_70[2];
    size_70 = None
    floordiv_15 = getitem_264 // 16;
    getitem_264 = None
    view_64 = bmm_15.view(floordiv_15, 16, getitem_265, 64);
    bmm_15 = None
    permute_31 = view_64.permute(0, 2, 1, 3);
    view_64 = None
    reshape_64 = permute_31.reshape(floordiv_15, getitem_265, 1024);
    permute_31 = floordiv_15 = getitem_265 = None
    transformer_h_15_self_attention_dense = getattr(self.transformer.h, "15").self_attention.dense(reshape_64);
    reshape_64 = None
    dropout_30 = torch.nn.functional.dropout(transformer_h_15_self_attention_dense, p=0.0, training=False,
                                             inplace=False);
    transformer_h_15_self_attention_dense = None
    add_63 = add_62 + dropout_30;
    add_62 = dropout_30 = None
    transformer_h_15_post_attention_layernorm = getattr(self.transformer.h, "15").post_attention_layernorm(add_63)
    transformer_h_15_mlp_dense_h_to_4h = getattr(self.transformer.h, "15").mlp.dense_h_to_4h(
        transformer_h_15_post_attention_layernorm);
    transformer_h_15_post_attention_layernorm = None
    mul_157 = transformer_h_15_mlp_dense_h_to_4h * 0.5
    mul_158 = 0.79788456 * transformer_h_15_mlp_dense_h_to_4h
    mul_159 = 0.044715 * transformer_h_15_mlp_dense_h_to_4h
    mul_160 = mul_159 * transformer_h_15_mlp_dense_h_to_4h;
    mul_159 = transformer_h_15_mlp_dense_h_to_4h = None
    add_64 = 1 + mul_160;
    mul_160 = None
    mul_161 = mul_158 * add_64;
    mul_158 = add_64 = None
    tanh_15 = torch.tanh(mul_161);
    mul_161 = None
    add_65 = 1.0 + tanh_15;
    tanh_15 = None
    mul_162 = mul_157 * add_65;
    mul_157 = add_65 = None
    transformer_h_15_mlp_dense_4h_to_h = getattr(self.transformer.h, "15").mlp.dense_4h_to_h(mul_162);
    mul_162 = None
    dropout_31 = torch.nn.functional.dropout(transformer_h_15_mlp_dense_4h_to_h, p=0.0, training=False, inplace=False);
    transformer_h_15_mlp_dense_4h_to_h = None
    add_66 = add_63 + dropout_31;
    add_63 = dropout_31 = None
    transformer_h_16_input_layernorm = getattr(self.transformer.h, "16").input_layernorm(add_66)
    transformer_h_16_self_attention_query_key_value = getattr(self.transformer.h, "16").self_attention.query_key_value(
        transformer_h_16_input_layernorm);
    transformer_h_16_input_layernorm = None
    size_71 = transformer_h_16_self_attention_query_key_value.size()
    getitem_267 = size_71[0]
    getitem_268 = size_71[1]
    getitem_269 = size_71[2];
    size_71 = None
    view_65 = transformer_h_16_self_attention_query_key_value.view(getitem_267, getitem_268, 16, 3, 64);
    transformer_h_16_self_attention_query_key_value = getitem_267 = getitem_268 = None
    getitem_270 = view_65[(Ellipsis, 0, slice(None, None, None))]
    getitem_271 = view_65[(Ellipsis, 1, slice(None, None, None))]
    getitem_272 = view_65[(Ellipsis, 2, slice(None, None, None))];
    view_65 = None
    size_72 = getitem_270.size()
    getitem_273 = size_72[0]
    getitem_274 = size_72[1]
    getitem_275 = size_72[2]
    getitem_276 = size_72[3];
    size_72 = None
    transpose_32 = getitem_270.transpose(1, 2);
    getitem_270 = None
    mul_163 = getitem_273 * 16
    reshape_65 = transpose_32.reshape(mul_163, getitem_274, 64);
    transpose_32 = mul_163 = None
    permute_32 = getitem_271.permute(0, 2, 3, 1);
    getitem_271 = None
    mul_164 = getitem_273 * 16
    reshape_66 = permute_32.reshape(mul_164, 64, getitem_274);
    permute_32 = mul_164 = None
    transpose_33 = getitem_272.transpose(1, 2);
    getitem_272 = None
    mul_165 = getitem_273 * 16
    reshape_67 = transpose_33.reshape(mul_165, getitem_274, 64);
    transpose_33 = mul_165 = None
    size_73 = reshape_66.size()
    getitem_277 = size_73[0]
    getitem_278 = size_73[1]
    getitem_279 = size_73[2];
    size_73 = None
    baddbmm_16 = to.baddbmm(batch1=reshape_65, batch2=reshape_66, beta=1.0, alpha=0.125);
    reshape_65 = reshape_66 = None
    view_66 = baddbmm_16.view(getitem_273, 16, getitem_274, getitem_279);
    baddbmm_16 = None
    getattr_59 = view_66.dtype
    eq_16 = getattr_59 == torch.float16
    getattr_60 = view_66.dtype
    finfo_19 = torch.finfo(getattr_60);
    getattr_60 = None
    getattr_61 = finfo_19.min;
    finfo_19 = None
    masked_fill_18 = torch.masked_fill(view_66, bool_2, getattr_61);
    view_66 = getattr_61 = None
    softmax_16 = torch.nn.functional.softmax(masked_fill_18, dim=-1, _stacklevel=3, dtype=torch.float32);
    masked_fill_18 = None
    to_21 = softmax_16.to(getattr_59);
    softmax_16 = getattr_59 = None
    transformer_h_16_self_attention_attention_dropout = getattr(self.transformer.h,
                                                                "16").self_attention.attention_dropout(to_21);
    to_21 = None
    mul_166 = getitem_273 * 16;
    getitem_273 = None
    view_67 = transformer_h_16_self_attention_attention_dropout.view(mul_166, getitem_274, getitem_279);
    transformer_h_16_self_attention_attention_dropout = mul_166 = getitem_274 = getitem_279 = None
    bmm_16 = torch.bmm(view_67, reshape_67);
    view_67 = reshape_67 = None
    size_74 = bmm_16.size()
    getitem_280 = size_74[0]
    getitem_281 = size_74[1]
    getitem_282 = size_74[2];
    size_74 = None
    floordiv_16 = getitem_280 // 16;
    getitem_280 = None
    view_68 = bmm_16.view(floordiv_16, 16, getitem_281, 64);
    bmm_16 = None
    permute_33 = view_68.permute(0, 2, 1, 3);
    view_68 = None
    reshape_68 = permute_33.reshape(floordiv_16, getitem_281, 1024);
    permute_33 = floordiv_16 = getitem_281 = None
    transformer_h_16_self_attention_dense = getattr(self.transformer.h, "16").self_attention.dense(reshape_68);
    reshape_68 = None
    dropout_32 = torch.nn.functional.dropout(transformer_h_16_self_attention_dense, p=0.0, training=False,
                                             inplace=False);
    transformer_h_16_self_attention_dense = None
    add_67 = add_66 + dropout_32;
    add_66 = dropout_32 = None
    transformer_h_16_post_attention_layernorm = getattr(self.transformer.h, "16").post_attention_layernorm(add_67)
    transformer_h_16_mlp_dense_h_to_4h = getattr(self.transformer.h, "16").mlp.dense_h_to_4h(
        transformer_h_16_post_attention_layernorm);
    transformer_h_16_post_attention_layernorm = None
    mul_167 = transformer_h_16_mlp_dense_h_to_4h * 0.5
    mul_168 = 0.79788456 * transformer_h_16_mlp_dense_h_to_4h
    mul_169 = 0.044715 * transformer_h_16_mlp_dense_h_to_4h
    mul_170 = mul_169 * transformer_h_16_mlp_dense_h_to_4h;
    mul_169 = transformer_h_16_mlp_dense_h_to_4h = None
    add_68 = 1 + mul_170;
    mul_170 = None
    mul_171 = mul_168 * add_68;
    mul_168 = add_68 = None
    tanh_16 = torch.tanh(mul_171);
    mul_171 = None
    add_69 = 1.0 + tanh_16;
    tanh_16 = None
    mul_172 = mul_167 * add_69;
    mul_167 = add_69 = None
    transformer_h_16_mlp_dense_4h_to_h = getattr(self.transformer.h, "16").mlp.dense_4h_to_h(mul_172);
    mul_172 = None
    dropout_33 = torch.nn.functional.dropout(transformer_h_16_mlp_dense_4h_to_h, p=0.0, training=False, inplace=False);
    transformer_h_16_mlp_dense_4h_to_h = None
    add_70 = add_67 + dropout_33;
    add_67 = dropout_33 = None
    transformer_h_17_input_layernorm = getattr(self.transformer.h, "17").input_layernorm(add_70)
    transformer_h_17_self_attention_query_key_value = getattr(self.transformer.h, "17").self_attention.query_key_value(
        transformer_h_17_input_layernorm);
    transformer_h_17_input_layernorm = None
    size_75 = transformer_h_17_self_attention_query_key_value.size()
    getitem_283 = size_75[0]
    getitem_284 = size_75[1]
    getitem_285 = size_75[2];
    size_75 = None
    view_69 = transformer_h_17_self_attention_query_key_value.view(getitem_283, getitem_284, 16, 3, 64);
    transformer_h_17_self_attention_query_key_value = getitem_283 = getitem_284 = None
    getitem_286 = view_69[(Ellipsis, 0, slice(None, None, None))]
    getitem_287 = view_69[(Ellipsis, 1, slice(None, None, None))]
    getitem_288 = view_69[(Ellipsis, 2, slice(None, None, None))];
    view_69 = None
    size_76 = getitem_286.size()
    getitem_289 = size_76[0]
    getitem_290 = size_76[1]
    getitem_291 = size_76[2]
    getitem_292 = size_76[3];
    size_76 = None
    transpose_34 = getitem_286.transpose(1, 2);
    getitem_286 = None
    mul_173 = getitem_289 * 16
    reshape_69 = transpose_34.reshape(mul_173, getitem_290, 64);
    transpose_34 = mul_173 = None
    permute_34 = getitem_287.permute(0, 2, 3, 1);
    getitem_287 = None
    mul_174 = getitem_289 * 16
    reshape_70 = permute_34.reshape(mul_174, 64, getitem_290);
    permute_34 = mul_174 = None
    transpose_35 = getitem_288.transpose(1, 2);
    getitem_288 = None
    mul_175 = getitem_289 * 16
    reshape_71 = transpose_35.reshape(mul_175, getitem_290, 64);
    transpose_35 = mul_175 = None
    size_77 = reshape_70.size()
    getitem_293 = size_77[0]
    getitem_294 = size_77[1]
    getitem_295 = size_77[2];
    size_77 = None
    baddbmm_17 = to.baddbmm(batch1=reshape_69, batch2=reshape_70, beta=1.0, alpha=0.125);
    reshape_69 = reshape_70 = None
    view_70 = baddbmm_17.view(getitem_289, 16, getitem_290, getitem_295);
    baddbmm_17 = None
    getattr_62 = view_70.dtype
    eq_17 = getattr_62 == torch.float16
    getattr_63 = view_70.dtype
    finfo_20 = torch.finfo(getattr_63);
    getattr_63 = None
    getattr_64 = finfo_20.min;
    finfo_20 = None
    masked_fill_19 = torch.masked_fill(view_70, bool_2, getattr_64);
    view_70 = getattr_64 = None
    softmax_17 = torch.nn.functional.softmax(masked_fill_19, dim=-1, _stacklevel=3, dtype=torch.float32);
    masked_fill_19 = None
    to_22 = softmax_17.to(getattr_62);
    softmax_17 = getattr_62 = None
    transformer_h_17_self_attention_attention_dropout = getattr(self.transformer.h,
                                                                "17").self_attention.attention_dropout(to_22);
    to_22 = None
    mul_176 = getitem_289 * 16;
    getitem_289 = None
    view_71 = transformer_h_17_self_attention_attention_dropout.view(mul_176, getitem_290, getitem_295);
    transformer_h_17_self_attention_attention_dropout = mul_176 = getitem_290 = getitem_295 = None
    bmm_17 = torch.bmm(view_71, reshape_71);
    view_71 = reshape_71 = None
    size_78 = bmm_17.size()
    getitem_296 = size_78[0]
    getitem_297 = size_78[1]
    getitem_298 = size_78[2];
    size_78 = None
    floordiv_17 = getitem_296 // 16;
    getitem_296 = None
    view_72 = bmm_17.view(floordiv_17, 16, getitem_297, 64);
    bmm_17 = None
    permute_35 = view_72.permute(0, 2, 1, 3);
    view_72 = None
    reshape_72 = permute_35.reshape(floordiv_17, getitem_297, 1024);
    permute_35 = floordiv_17 = getitem_297 = None
    transformer_h_17_self_attention_dense = getattr(self.transformer.h, "17").self_attention.dense(reshape_72);
    reshape_72 = None
    dropout_34 = torch.nn.functional.dropout(transformer_h_17_self_attention_dense, p=0.0, training=False,
                                             inplace=False);
    transformer_h_17_self_attention_dense = None
    add_71 = add_70 + dropout_34;
    add_70 = dropout_34 = None
    transformer_h_17_post_attention_layernorm = getattr(self.transformer.h, "17").post_attention_layernorm(add_71)
    transformer_h_17_mlp_dense_h_to_4h = getattr(self.transformer.h, "17").mlp.dense_h_to_4h(
        transformer_h_17_post_attention_layernorm);
    transformer_h_17_post_attention_layernorm = None
    mul_177 = transformer_h_17_mlp_dense_h_to_4h * 0.5
    mul_178 = 0.79788456 * transformer_h_17_mlp_dense_h_to_4h
    mul_179 = 0.044715 * transformer_h_17_mlp_dense_h_to_4h
    mul_180 = mul_179 * transformer_h_17_mlp_dense_h_to_4h;
    mul_179 = transformer_h_17_mlp_dense_h_to_4h = None
    add_72 = 1 + mul_180;
    mul_180 = None
    mul_181 = mul_178 * add_72;
    mul_178 = add_72 = None
    tanh_17 = torch.tanh(mul_181);
    mul_181 = None
    add_73 = 1.0 + tanh_17;
    tanh_17 = None
    mul_182 = mul_177 * add_73;
    mul_177 = add_73 = None
    transformer_h_17_mlp_dense_4h_to_h = getattr(self.transformer.h, "17").mlp.dense_4h_to_h(mul_182);
    mul_182 = None
    dropout_35 = torch.nn.functional.dropout(transformer_h_17_mlp_dense_4h_to_h, p=0.0, training=False, inplace=False);
    transformer_h_17_mlp_dense_4h_to_h = None
    add_74 = add_71 + dropout_35;
    add_71 = dropout_35 = None
    transformer_h_18_input_layernorm = getattr(self.transformer.h, "18").input_layernorm(add_74)
    transformer_h_18_self_attention_query_key_value = getattr(self.transformer.h, "18").self_attention.query_key_value(
        transformer_h_18_input_layernorm);
    transformer_h_18_input_layernorm = None
    size_79 = transformer_h_18_self_attention_query_key_value.size()
    getitem_299 = size_79[0]
    getitem_300 = size_79[1]
    getitem_301 = size_79[2];
    size_79 = None
    view_73 = transformer_h_18_self_attention_query_key_value.view(getitem_299, getitem_300, 16, 3, 64);
    transformer_h_18_self_attention_query_key_value = getitem_299 = getitem_300 = None
    getitem_302 = view_73[(Ellipsis, 0, slice(None, None, None))]
    getitem_303 = view_73[(Ellipsis, 1, slice(None, None, None))]
    getitem_304 = view_73[(Ellipsis, 2, slice(None, None, None))];
    view_73 = None
    size_80 = getitem_302.size()
    getitem_305 = size_80[0]
    getitem_306 = size_80[1]
    getitem_307 = size_80[2]
    getitem_308 = size_80[3];
    size_80 = None
    transpose_36 = getitem_302.transpose(1, 2);
    getitem_302 = None
    mul_183 = getitem_305 * 16
    reshape_73 = transpose_36.reshape(mul_183, getitem_306, 64);
    transpose_36 = mul_183 = None
    permute_36 = getitem_303.permute(0, 2, 3, 1);
    getitem_303 = None
    mul_184 = getitem_305 * 16
    reshape_74 = permute_36.reshape(mul_184, 64, getitem_306);
    permute_36 = mul_184 = None
    transpose_37 = getitem_304.transpose(1, 2);
    getitem_304 = None
    mul_185 = getitem_305 * 16
    reshape_75 = transpose_37.reshape(mul_185, getitem_306, 64);
    transpose_37 = mul_185 = None
    size_81 = reshape_74.size()
    getitem_309 = size_81[0]
    getitem_310 = size_81[1]
    getitem_311 = size_81[2];
    size_81 = None
    baddbmm_18 = to.baddbmm(batch1=reshape_73, batch2=reshape_74, beta=1.0, alpha=0.125);
    reshape_73 = reshape_74 = None
    view_74 = baddbmm_18.view(getitem_305, 16, getitem_306, getitem_311);
    baddbmm_18 = None
    getattr_65 = view_74.dtype
    eq_18 = getattr_65 == torch.float16
    getattr_66 = view_74.dtype
    finfo_21 = torch.finfo(getattr_66);
    getattr_66 = None
    getattr_67 = finfo_21.min;
    finfo_21 = None
    masked_fill_20 = torch.masked_fill(view_74, bool_2, getattr_67);
    view_74 = getattr_67 = None
    softmax_18 = torch.nn.functional.softmax(masked_fill_20, dim=-1, _stacklevel=3, dtype=torch.float32);
    masked_fill_20 = None
    to_23 = softmax_18.to(getattr_65);
    softmax_18 = getattr_65 = None
    transformer_h_18_self_attention_attention_dropout = getattr(self.transformer.h,
                                                                "18").self_attention.attention_dropout(to_23);
    to_23 = None
    mul_186 = getitem_305 * 16;
    getitem_305 = None
    view_75 = transformer_h_18_self_attention_attention_dropout.view(mul_186, getitem_306, getitem_311);
    transformer_h_18_self_attention_attention_dropout = mul_186 = getitem_306 = getitem_311 = None
    bmm_18 = torch.bmm(view_75, reshape_75);
    view_75 = reshape_75 = None
    size_82 = bmm_18.size()
    getitem_312 = size_82[0]
    getitem_313 = size_82[1]
    getitem_314 = size_82[2];
    size_82 = None
    floordiv_18 = getitem_312 // 16;
    getitem_312 = None
    view_76 = bmm_18.view(floordiv_18, 16, getitem_313, 64);
    bmm_18 = None
    permute_37 = view_76.permute(0, 2, 1, 3);
    view_76 = None
    reshape_76 = permute_37.reshape(floordiv_18, getitem_313, 1024);
    permute_37 = floordiv_18 = getitem_313 = None
    transformer_h_18_self_attention_dense = getattr(self.transformer.h, "18").self_attention.dense(reshape_76);
    reshape_76 = None
    dropout_36 = torch.nn.functional.dropout(transformer_h_18_self_attention_dense, p=0.0, training=False,
                                             inplace=False);
    transformer_h_18_self_attention_dense = None
    add_75 = add_74 + dropout_36;
    add_74 = dropout_36 = None
    transformer_h_18_post_attention_layernorm = getattr(self.transformer.h, "18").post_attention_layernorm(add_75)
    transformer_h_18_mlp_dense_h_to_4h = getattr(self.transformer.h, "18").mlp.dense_h_to_4h(
        transformer_h_18_post_attention_layernorm);
    transformer_h_18_post_attention_layernorm = None
    mul_187 = transformer_h_18_mlp_dense_h_to_4h * 0.5
    mul_188 = 0.79788456 * transformer_h_18_mlp_dense_h_to_4h
    mul_189 = 0.044715 * transformer_h_18_mlp_dense_h_to_4h
    mul_190 = mul_189 * transformer_h_18_mlp_dense_h_to_4h;
    mul_189 = transformer_h_18_mlp_dense_h_to_4h = None
    add_76 = 1 + mul_190;
    mul_190 = None
    mul_191 = mul_188 * add_76;
    mul_188 = add_76 = None
    tanh_18 = torch.tanh(mul_191);
    mul_191 = None
    add_77 = 1.0 + tanh_18;
    tanh_18 = None
    mul_192 = mul_187 * add_77;
    mul_187 = add_77 = None
    transformer_h_18_mlp_dense_4h_to_h = getattr(self.transformer.h, "18").mlp.dense_4h_to_h(mul_192);
    mul_192 = None
    dropout_37 = torch.nn.functional.dropout(transformer_h_18_mlp_dense_4h_to_h, p=0.0, training=False, inplace=False);
    transformer_h_18_mlp_dense_4h_to_h = None
    add_78 = add_75 + dropout_37;
    add_75 = dropout_37 = None
    transformer_h_19_input_layernorm = getattr(self.transformer.h, "19").input_layernorm(add_78)
    transformer_h_19_self_attention_query_key_value = getattr(self.transformer.h, "19").self_attention.query_key_value(
        transformer_h_19_input_layernorm);
    transformer_h_19_input_layernorm = None
    size_83 = transformer_h_19_self_attention_query_key_value.size()
    getitem_315 = size_83[0]
    getitem_316 = size_83[1]
    getitem_317 = size_83[2];
    size_83 = None
    view_77 = transformer_h_19_self_attention_query_key_value.view(getitem_315, getitem_316, 16, 3, 64);
    transformer_h_19_self_attention_query_key_value = getitem_315 = getitem_316 = None
    getitem_318 = view_77[(Ellipsis, 0, slice(None, None, None))]
    getitem_319 = view_77[(Ellipsis, 1, slice(None, None, None))]
    getitem_320 = view_77[(Ellipsis, 2, slice(None, None, None))];
    view_77 = None
    size_84 = getitem_318.size()
    getitem_321 = size_84[0]
    getitem_322 = size_84[1]
    getitem_323 = size_84[2]
    getitem_324 = size_84[3];
    size_84 = None
    transpose_38 = getitem_318.transpose(1, 2);
    getitem_318 = None
    mul_193 = getitem_321 * 16
    reshape_77 = transpose_38.reshape(mul_193, getitem_322, 64);
    transpose_38 = mul_193 = None
    permute_38 = getitem_319.permute(0, 2, 3, 1);
    getitem_319 = None
    mul_194 = getitem_321 * 16
    reshape_78 = permute_38.reshape(mul_194, 64, getitem_322);
    permute_38 = mul_194 = None
    transpose_39 = getitem_320.transpose(1, 2);
    getitem_320 = None
    mul_195 = getitem_321 * 16
    reshape_79 = transpose_39.reshape(mul_195, getitem_322, 64);
    transpose_39 = mul_195 = None
    size_85 = reshape_78.size()
    getitem_325 = size_85[0]
    getitem_326 = size_85[1]
    getitem_327 = size_85[2];
    size_85 = None
    baddbmm_19 = to.baddbmm(batch1=reshape_77, batch2=reshape_78, beta=1.0, alpha=0.125);
    reshape_77 = reshape_78 = None
    view_78 = baddbmm_19.view(getitem_321, 16, getitem_322, getitem_327);
    baddbmm_19 = None
    getattr_68 = view_78.dtype
    eq_19 = getattr_68 == torch.float16
    getattr_69 = view_78.dtype
    finfo_22 = torch.finfo(getattr_69);
    getattr_69 = None
    getattr_70 = finfo_22.min;
    finfo_22 = None
    masked_fill_21 = torch.masked_fill(view_78, bool_2, getattr_70);
    view_78 = getattr_70 = None
    softmax_19 = torch.nn.functional.softmax(masked_fill_21, dim=-1, _stacklevel=3, dtype=torch.float32);
    masked_fill_21 = None
    to_24 = softmax_19.to(getattr_68);
    softmax_19 = getattr_68 = None
    transformer_h_19_self_attention_attention_dropout = getattr(self.transformer.h,
                                                                "19").self_attention.attention_dropout(to_24);
    to_24 = None
    mul_196 = getitem_321 * 16;
    getitem_321 = None
    view_79 = transformer_h_19_self_attention_attention_dropout.view(mul_196, getitem_322, getitem_327);
    transformer_h_19_self_attention_attention_dropout = mul_196 = getitem_322 = getitem_327 = None
    bmm_19 = torch.bmm(view_79, reshape_79);
    view_79 = reshape_79 = None
    size_86 = bmm_19.size()
    getitem_328 = size_86[0]
    getitem_329 = size_86[1]
    getitem_330 = size_86[2];
    size_86 = None
    floordiv_19 = getitem_328 // 16;
    getitem_328 = None
    view_80 = bmm_19.view(floordiv_19, 16, getitem_329, 64);
    bmm_19 = None
    permute_39 = view_80.permute(0, 2, 1, 3);
    view_80 = None
    reshape_80 = permute_39.reshape(floordiv_19, getitem_329, 1024);
    permute_39 = floordiv_19 = getitem_329 = None
    transformer_h_19_self_attention_dense = getattr(self.transformer.h, "19").self_attention.dense(reshape_80);
    reshape_80 = None
    dropout_38 = torch.nn.functional.dropout(transformer_h_19_self_attention_dense, p=0.0, training=False,
                                             inplace=False);
    transformer_h_19_self_attention_dense = None
    add_79 = add_78 + dropout_38;
    add_78 = dropout_38 = None
    transformer_h_19_post_attention_layernorm = getattr(self.transformer.h, "19").post_attention_layernorm(add_79)
    transformer_h_19_mlp_dense_h_to_4h = getattr(self.transformer.h, "19").mlp.dense_h_to_4h(
        transformer_h_19_post_attention_layernorm);
    transformer_h_19_post_attention_layernorm = None
    mul_197 = transformer_h_19_mlp_dense_h_to_4h * 0.5
    mul_198 = 0.79788456 * transformer_h_19_mlp_dense_h_to_4h
    mul_199 = 0.044715 * transformer_h_19_mlp_dense_h_to_4h
    mul_200 = mul_199 * transformer_h_19_mlp_dense_h_to_4h;
    mul_199 = transformer_h_19_mlp_dense_h_to_4h = None
    add_80 = 1 + mul_200;
    mul_200 = None
    mul_201 = mul_198 * add_80;
    mul_198 = add_80 = None
    tanh_19 = torch.tanh(mul_201);
    mul_201 = None
    add_81 = 1.0 + tanh_19;
    tanh_19 = None
    mul_202 = mul_197 * add_81;
    mul_197 = add_81 = None
    transformer_h_19_mlp_dense_4h_to_h = getattr(self.transformer.h, "19").mlp.dense_4h_to_h(mul_202);
    mul_202 = None
    dropout_39 = torch.nn.functional.dropout(transformer_h_19_mlp_dense_4h_to_h, p=0.0, training=False, inplace=False);
    transformer_h_19_mlp_dense_4h_to_h = None
    add_82 = add_79 + dropout_39;
    add_79 = dropout_39 = None
    transformer_h_20_input_layernorm = getattr(self.transformer.h, "20").input_layernorm(add_82)
    transformer_h_20_self_attention_query_key_value = getattr(self.transformer.h, "20").self_attention.query_key_value(
        transformer_h_20_input_layernorm);
    transformer_h_20_input_layernorm = None
    size_87 = transformer_h_20_self_attention_query_key_value.size()
    getitem_331 = size_87[0]
    getitem_332 = size_87[1]
    getitem_333 = size_87[2];
    size_87 = None
    view_81 = transformer_h_20_self_attention_query_key_value.view(getitem_331, getitem_332, 16, 3, 64);
    transformer_h_20_self_attention_query_key_value = getitem_331 = getitem_332 = None
    getitem_334 = view_81[(Ellipsis, 0, slice(None, None, None))]
    getitem_335 = view_81[(Ellipsis, 1, slice(None, None, None))]
    getitem_336 = view_81[(Ellipsis, 2, slice(None, None, None))];
    view_81 = None
    size_88 = getitem_334.size()
    getitem_337 = size_88[0]
    getitem_338 = size_88[1]
    getitem_339 = size_88[2]
    getitem_340 = size_88[3];
    size_88 = None
    transpose_40 = getitem_334.transpose(1, 2);
    getitem_334 = None
    mul_203 = getitem_337 * 16
    reshape_81 = transpose_40.reshape(mul_203, getitem_338, 64);
    transpose_40 = mul_203 = None
    permute_40 = getitem_335.permute(0, 2, 3, 1);
    getitem_335 = None
    mul_204 = getitem_337 * 16
    reshape_82 = permute_40.reshape(mul_204, 64, getitem_338);
    permute_40 = mul_204 = None
    transpose_41 = getitem_336.transpose(1, 2);
    getitem_336 = None
    mul_205 = getitem_337 * 16
    reshape_83 = transpose_41.reshape(mul_205, getitem_338, 64);
    transpose_41 = mul_205 = None
    size_89 = reshape_82.size()
    getitem_341 = size_89[0]
    getitem_342 = size_89[1]
    getitem_343 = size_89[2];
    size_89 = None
    baddbmm_20 = to.baddbmm(batch1=reshape_81, batch2=reshape_82, beta=1.0, alpha=0.125);
    reshape_81 = reshape_82 = None
    view_82 = baddbmm_20.view(getitem_337, 16, getitem_338, getitem_343);
    baddbmm_20 = None
    getattr_71 = view_82.dtype
    eq_20 = getattr_71 == torch.float16
    getattr_72 = view_82.dtype
    finfo_23 = torch.finfo(getattr_72);
    getattr_72 = None
    getattr_73 = finfo_23.min;
    finfo_23 = None
    masked_fill_22 = torch.masked_fill(view_82, bool_2, getattr_73);
    view_82 = getattr_73 = None
    softmax_20 = torch.nn.functional.softmax(masked_fill_22, dim=-1, _stacklevel=3, dtype=torch.float32);
    masked_fill_22 = None
    to_25 = softmax_20.to(getattr_71);
    softmax_20 = getattr_71 = None
    transformer_h_20_self_attention_attention_dropout = getattr(self.transformer.h,
                                                                "20").self_attention.attention_dropout(to_25);
    to_25 = None
    mul_206 = getitem_337 * 16;
    getitem_337 = None
    view_83 = transformer_h_20_self_attention_attention_dropout.view(mul_206, getitem_338, getitem_343);
    transformer_h_20_self_attention_attention_dropout = mul_206 = getitem_338 = getitem_343 = None
    bmm_20 = torch.bmm(view_83, reshape_83);
    view_83 = reshape_83 = None
    size_90 = bmm_20.size()
    getitem_344 = size_90[0]
    getitem_345 = size_90[1]
    getitem_346 = size_90[2];
    size_90 = None
    floordiv_20 = getitem_344 // 16;
    getitem_344 = None
    view_84 = bmm_20.view(floordiv_20, 16, getitem_345, 64);
    bmm_20 = None
    permute_41 = view_84.permute(0, 2, 1, 3);
    view_84 = None
    reshape_84 = permute_41.reshape(floordiv_20, getitem_345, 1024);
    permute_41 = floordiv_20 = getitem_345 = None
    transformer_h_20_self_attention_dense = getattr(self.transformer.h, "20").self_attention.dense(reshape_84);
    reshape_84 = None
    dropout_40 = torch.nn.functional.dropout(transformer_h_20_self_attention_dense, p=0.0, training=False,
                                             inplace=False);
    transformer_h_20_self_attention_dense = None
    add_83 = add_82 + dropout_40;
    add_82 = dropout_40 = None
    transformer_h_20_post_attention_layernorm = getattr(self.transformer.h, "20").post_attention_layernorm(add_83)
    transformer_h_20_mlp_dense_h_to_4h = getattr(self.transformer.h, "20").mlp.dense_h_to_4h(
        transformer_h_20_post_attention_layernorm);
    transformer_h_20_post_attention_layernorm = None
    mul_207 = transformer_h_20_mlp_dense_h_to_4h * 0.5
    mul_208 = 0.79788456 * transformer_h_20_mlp_dense_h_to_4h
    mul_209 = 0.044715 * transformer_h_20_mlp_dense_h_to_4h
    mul_210 = mul_209 * transformer_h_20_mlp_dense_h_to_4h;
    mul_209 = transformer_h_20_mlp_dense_h_to_4h = None
    add_84 = 1 + mul_210;
    mul_210 = None
    mul_211 = mul_208 * add_84;
    mul_208 = add_84 = None
    tanh_20 = torch.tanh(mul_211);
    mul_211 = None
    add_85 = 1.0 + tanh_20;
    tanh_20 = None
    mul_212 = mul_207 * add_85;
    mul_207 = add_85 = None
    transformer_h_20_mlp_dense_4h_to_h = getattr(self.transformer.h, "20").mlp.dense_4h_to_h(mul_212);
    mul_212 = None
    dropout_41 = torch.nn.functional.dropout(transformer_h_20_mlp_dense_4h_to_h, p=0.0, training=False, inplace=False);
    transformer_h_20_mlp_dense_4h_to_h = None
    add_86 = add_83 + dropout_41;
    add_83 = dropout_41 = None
    transformer_h_21_input_layernorm = getattr(self.transformer.h, "21").input_layernorm(add_86)
    transformer_h_21_self_attention_query_key_value = getattr(self.transformer.h, "21").self_attention.query_key_value(
        transformer_h_21_input_layernorm);
    transformer_h_21_input_layernorm = None
    size_91 = transformer_h_21_self_attention_query_key_value.size()
    getitem_347 = size_91[0]
    getitem_348 = size_91[1]
    getitem_349 = size_91[2];
    size_91 = None
    view_85 = transformer_h_21_self_attention_query_key_value.view(getitem_347, getitem_348, 16, 3, 64);
    transformer_h_21_self_attention_query_key_value = getitem_347 = getitem_348 = None
    getitem_350 = view_85[(Ellipsis, 0, slice(None, None, None))]
    getitem_351 = view_85[(Ellipsis, 1, slice(None, None, None))]
    getitem_352 = view_85[(Ellipsis, 2, slice(None, None, None))];
    view_85 = None
    size_92 = getitem_350.size()
    getitem_353 = size_92[0]
    getitem_354 = size_92[1]
    getitem_355 = size_92[2]
    getitem_356 = size_92[3];
    size_92 = None
    transpose_42 = getitem_350.transpose(1, 2);
    getitem_350 = None
    mul_213 = getitem_353 * 16
    reshape_85 = transpose_42.reshape(mul_213, getitem_354, 64);
    transpose_42 = mul_213 = None
    permute_42 = getitem_351.permute(0, 2, 3, 1);
    getitem_351 = None
    mul_214 = getitem_353 * 16
    reshape_86 = permute_42.reshape(mul_214, 64, getitem_354);
    permute_42 = mul_214 = None
    transpose_43 = getitem_352.transpose(1, 2);
    getitem_352 = None
    mul_215 = getitem_353 * 16
    reshape_87 = transpose_43.reshape(mul_215, getitem_354, 64);
    transpose_43 = mul_215 = None
    size_93 = reshape_86.size()
    getitem_357 = size_93[0]
    getitem_358 = size_93[1]
    getitem_359 = size_93[2];
    size_93 = None
    baddbmm_21 = to.baddbmm(batch1=reshape_85, batch2=reshape_86, beta=1.0, alpha=0.125);
    reshape_85 = reshape_86 = None
    view_86 = baddbmm_21.view(getitem_353, 16, getitem_354, getitem_359);
    baddbmm_21 = None
    getattr_74 = view_86.dtype
    eq_21 = getattr_74 == torch.float16
    getattr_75 = view_86.dtype
    finfo_24 = torch.finfo(getattr_75);
    getattr_75 = None
    getattr_76 = finfo_24.min;
    finfo_24 = None
    masked_fill_23 = torch.masked_fill(view_86, bool_2, getattr_76);
    view_86 = getattr_76 = None
    softmax_21 = torch.nn.functional.softmax(masked_fill_23, dim=-1, _stacklevel=3, dtype=torch.float32);
    masked_fill_23 = None
    to_26 = softmax_21.to(getattr_74);
    softmax_21 = getattr_74 = None
    transformer_h_21_self_attention_attention_dropout = getattr(self.transformer.h,
                                                                "21").self_attention.attention_dropout(to_26);
    to_26 = None
    mul_216 = getitem_353 * 16;
    getitem_353 = None
    view_87 = transformer_h_21_self_attention_attention_dropout.view(mul_216, getitem_354, getitem_359);
    transformer_h_21_self_attention_attention_dropout = mul_216 = getitem_354 = getitem_359 = None
    bmm_21 = torch.bmm(view_87, reshape_87);
    view_87 = reshape_87 = None
    size_94 = bmm_21.size()
    getitem_360 = size_94[0]
    getitem_361 = size_94[1]
    getitem_362 = size_94[2];
    size_94 = None
    floordiv_21 = getitem_360 // 16;
    getitem_360 = None
    view_88 = bmm_21.view(floordiv_21, 16, getitem_361, 64);
    bmm_21 = None
    permute_43 = view_88.permute(0, 2, 1, 3);
    view_88 = None
    reshape_88 = permute_43.reshape(floordiv_21, getitem_361, 1024);
    permute_43 = floordiv_21 = getitem_361 = None
    transformer_h_21_self_attention_dense = getattr(self.transformer.h, "21").self_attention.dense(reshape_88);
    reshape_88 = None
    dropout_42 = torch.nn.functional.dropout(transformer_h_21_self_attention_dense, p=0.0, training=False,
                                             inplace=False);
    transformer_h_21_self_attention_dense = None
    add_87 = add_86 + dropout_42;
    add_86 = dropout_42 = None
    transformer_h_21_post_attention_layernorm = getattr(self.transformer.h, "21").post_attention_layernorm(add_87)
    transformer_h_21_mlp_dense_h_to_4h = getattr(self.transformer.h, "21").mlp.dense_h_to_4h(
        transformer_h_21_post_attention_layernorm);
    transformer_h_21_post_attention_layernorm = None
    mul_217 = transformer_h_21_mlp_dense_h_to_4h * 0.5
    mul_218 = 0.79788456 * transformer_h_21_mlp_dense_h_to_4h
    mul_219 = 0.044715 * transformer_h_21_mlp_dense_h_to_4h
    mul_220 = mul_219 * transformer_h_21_mlp_dense_h_to_4h;
    mul_219 = transformer_h_21_mlp_dense_h_to_4h = None
    add_88 = 1 + mul_220;
    mul_220 = None
    mul_221 = mul_218 * add_88;
    mul_218 = add_88 = None
    tanh_21 = torch.tanh(mul_221);
    mul_221 = None
    add_89 = 1.0 + tanh_21;
    tanh_21 = None
    mul_222 = mul_217 * add_89;
    mul_217 = add_89 = None
    transformer_h_21_mlp_dense_4h_to_h = getattr(self.transformer.h, "21").mlp.dense_4h_to_h(mul_222);
    mul_222 = None
    dropout_43 = torch.nn.functional.dropout(transformer_h_21_mlp_dense_4h_to_h, p=0.0, training=False, inplace=False);
    transformer_h_21_mlp_dense_4h_to_h = None
    add_90 = add_87 + dropout_43;
    add_87 = dropout_43 = None
    transformer_h_22_input_layernorm = getattr(self.transformer.h, "22").input_layernorm(add_90)
    transformer_h_22_self_attention_query_key_value = getattr(self.transformer.h, "22").self_attention.query_key_value(
        transformer_h_22_input_layernorm);
    transformer_h_22_input_layernorm = None
    size_95 = transformer_h_22_self_attention_query_key_value.size()
    getitem_363 = size_95[0]
    getitem_364 = size_95[1]
    getitem_365 = size_95[2];
    size_95 = None
    view_89 = transformer_h_22_self_attention_query_key_value.view(getitem_363, getitem_364, 16, 3, 64);
    transformer_h_22_self_attention_query_key_value = getitem_363 = getitem_364 = None
    getitem_366 = view_89[(Ellipsis, 0, slice(None, None, None))]
    getitem_367 = view_89[(Ellipsis, 1, slice(None, None, None))]
    getitem_368 = view_89[(Ellipsis, 2, slice(None, None, None))];
    view_89 = None
    size_96 = getitem_366.size()
    getitem_369 = size_96[0]
    getitem_370 = size_96[1]
    getitem_371 = size_96[2]
    getitem_372 = size_96[3];
    size_96 = None
    transpose_44 = getitem_366.transpose(1, 2);
    getitem_366 = None
    mul_223 = getitem_369 * 16
    reshape_89 = transpose_44.reshape(mul_223, getitem_370, 64);
    transpose_44 = mul_223 = None
    permute_44 = getitem_367.permute(0, 2, 3, 1);
    getitem_367 = None
    mul_224 = getitem_369 * 16
    reshape_90 = permute_44.reshape(mul_224, 64, getitem_370);
    permute_44 = mul_224 = None
    transpose_45 = getitem_368.transpose(1, 2);
    getitem_368 = None
    mul_225 = getitem_369 * 16
    reshape_91 = transpose_45.reshape(mul_225, getitem_370, 64);
    transpose_45 = mul_225 = None
    size_97 = reshape_90.size()
    getitem_373 = size_97[0]
    getitem_374 = size_97[1]
    getitem_375 = size_97[2];
    size_97 = None
    baddbmm_22 = to.baddbmm(batch1=reshape_89, batch2=reshape_90, beta=1.0, alpha=0.125);
    reshape_89 = reshape_90 = None
    view_90 = baddbmm_22.view(getitem_369, 16, getitem_370, getitem_375);
    baddbmm_22 = None
    getattr_77 = view_90.dtype
    eq_22 = getattr_77 == torch.float16
    getattr_78 = view_90.dtype
    finfo_25 = torch.finfo(getattr_78);
    getattr_78 = None
    getattr_79 = finfo_25.min;
    finfo_25 = None
    masked_fill_24 = torch.masked_fill(view_90, bool_2, getattr_79);
    view_90 = getattr_79 = None
    softmax_22 = torch.nn.functional.softmax(masked_fill_24, dim=-1, _stacklevel=3, dtype=torch.float32);
    masked_fill_24 = None
    to_27 = softmax_22.to(getattr_77);
    softmax_22 = getattr_77 = None
    transformer_h_22_self_attention_attention_dropout = getattr(self.transformer.h,
                                                                "22").self_attention.attention_dropout(to_27);
    to_27 = None
    mul_226 = getitem_369 * 16;
    getitem_369 = None
    view_91 = transformer_h_22_self_attention_attention_dropout.view(mul_226, getitem_370, getitem_375);
    transformer_h_22_self_attention_attention_dropout = mul_226 = getitem_370 = getitem_375 = None
    bmm_22 = torch.bmm(view_91, reshape_91);
    view_91 = reshape_91 = None
    size_98 = bmm_22.size()
    getitem_376 = size_98[0]
    getitem_377 = size_98[1]
    getitem_378 = size_98[2];
    size_98 = None
    floordiv_22 = getitem_376 // 16;
    getitem_376 = None
    view_92 = bmm_22.view(floordiv_22, 16, getitem_377, 64);
    bmm_22 = None
    permute_45 = view_92.permute(0, 2, 1, 3);
    view_92 = None
    reshape_92 = permute_45.reshape(floordiv_22, getitem_377, 1024);
    permute_45 = floordiv_22 = getitem_377 = None
    transformer_h_22_self_attention_dense = getattr(self.transformer.h, "22").self_attention.dense(reshape_92);
    reshape_92 = None
    dropout_44 = torch.nn.functional.dropout(transformer_h_22_self_attention_dense, p=0.0, training=False,
                                             inplace=False);
    transformer_h_22_self_attention_dense = None
    add_91 = add_90 + dropout_44;
    add_90 = dropout_44 = None
    transformer_h_22_post_attention_layernorm = getattr(self.transformer.h, "22").post_attention_layernorm(add_91)
    transformer_h_22_mlp_dense_h_to_4h = getattr(self.transformer.h, "22").mlp.dense_h_to_4h(
        transformer_h_22_post_attention_layernorm);
    transformer_h_22_post_attention_layernorm = None
    mul_227 = transformer_h_22_mlp_dense_h_to_4h * 0.5
    mul_228 = 0.79788456 * transformer_h_22_mlp_dense_h_to_4h
    mul_229 = 0.044715 * transformer_h_22_mlp_dense_h_to_4h
    mul_230 = mul_229 * transformer_h_22_mlp_dense_h_to_4h;
    mul_229 = transformer_h_22_mlp_dense_h_to_4h = None
    add_92 = 1 + mul_230;
    mul_230 = None
    mul_231 = mul_228 * add_92;
    mul_228 = add_92 = None
    tanh_22 = torch.tanh(mul_231);
    mul_231 = None
    add_93 = 1.0 + tanh_22;
    tanh_22 = None
    mul_232 = mul_227 * add_93;
    mul_227 = add_93 = None
    transformer_h_22_mlp_dense_4h_to_h = getattr(self.transformer.h, "22").mlp.dense_4h_to_h(mul_232);
    mul_232 = None
    dropout_45 = torch.nn.functional.dropout(transformer_h_22_mlp_dense_4h_to_h, p=0.0, training=False, inplace=False);
    transformer_h_22_mlp_dense_4h_to_h = None
    add_94 = add_91 + dropout_45;
    add_91 = dropout_45 = None
    transformer_h_23_input_layernorm = getattr(self.transformer.h, "23").input_layernorm(add_94)
    transformer_h_23_self_attention_query_key_value = getattr(self.transformer.h, "23").self_attention.query_key_value(
        transformer_h_23_input_layernorm);
    transformer_h_23_input_layernorm = None
    size_99 = transformer_h_23_self_attention_query_key_value.size()
    getitem_379 = size_99[0]
    getitem_380 = size_99[1]
    getitem_381 = size_99[2];
    size_99 = None
    view_93 = transformer_h_23_self_attention_query_key_value.view(getitem_379, getitem_380, 16, 3, 64);
    transformer_h_23_self_attention_query_key_value = getitem_379 = getitem_380 = None
    getitem_382 = view_93[(Ellipsis, 0, slice(None, None, None))]
    getitem_383 = view_93[(Ellipsis, 1, slice(None, None, None))]
    getitem_384 = view_93[(Ellipsis, 2, slice(None, None, None))];
    view_93 = None
    size_100 = getitem_382.size()
    getitem_385 = size_100[0]
    getitem_386 = size_100[1]
    getitem_387 = size_100[2]
    getitem_388 = size_100[3];
    size_100 = None
    transpose_46 = getitem_382.transpose(1, 2);
    getitem_382 = None
    mul_233 = getitem_385 * 16
    reshape_93 = transpose_46.reshape(mul_233, getitem_386, 64);
    transpose_46 = mul_233 = None
    permute_46 = getitem_383.permute(0, 2, 3, 1);
    getitem_383 = None
    mul_234 = getitem_385 * 16
    reshape_94 = permute_46.reshape(mul_234, 64, getitem_386);
    permute_46 = mul_234 = None
    transpose_47 = getitem_384.transpose(1, 2);
    getitem_384 = None
    mul_235 = getitem_385 * 16
    reshape_95 = transpose_47.reshape(mul_235, getitem_386, 64);
    transpose_47 = mul_235 = None
    size_101 = reshape_94.size()
    getitem_389 = size_101[0]
    getitem_390 = size_101[1]
    getitem_391 = size_101[2];
    size_101 = None
    baddbmm_23 = to.baddbmm(batch1=reshape_93, batch2=reshape_94, beta=1.0, alpha=0.125);
    to = reshape_93 = reshape_94 = None
    view_94 = baddbmm_23.view(getitem_385, 16, getitem_386, getitem_391);
    baddbmm_23 = None
    getattr_80 = view_94.dtype
    eq_23 = getattr_80 == torch.float16
    getattr_81 = view_94.dtype
    finfo_26 = torch.finfo(getattr_81);
    getattr_81 = None
    getattr_82 = finfo_26.min;
    finfo_26 = None
    masked_fill_25 = torch.masked_fill(view_94, bool_2, getattr_82);
    view_94 = bool_2 = getattr_82 = None
    softmax_23 = torch.nn.functional.softmax(masked_fill_25, dim=-1, _stacklevel=3, dtype=torch.float32);
    masked_fill_25 = None
    to_28 = softmax_23.to(getattr_80);
    softmax_23 = getattr_80 = None
    transformer_h_23_self_attention_attention_dropout = getattr(self.transformer.h,
                                                                "23").self_attention.attention_dropout(to_28);
    to_28 = None
    mul_236 = getitem_385 * 16;
    getitem_385 = None
    view_95 = transformer_h_23_self_attention_attention_dropout.view(mul_236, getitem_386, getitem_391);
    transformer_h_23_self_attention_attention_dropout = mul_236 = getitem_386 = getitem_391 = None
    bmm_23 = torch.bmm(view_95, reshape_95);
    view_95 = reshape_95 = None
    size_102 = bmm_23.size()
    getitem_392 = size_102[0]
    getitem_393 = size_102[1]
    getitem_394 = size_102[2];
    size_102 = None
    floordiv_23 = getitem_392 // 16;
    getitem_392 = None
    view_96 = bmm_23.view(floordiv_23, 16, getitem_393, 64);
    bmm_23 = None
    permute_47 = view_96.permute(0, 2, 1, 3);
    view_96 = None
    reshape_96 = permute_47.reshape(floordiv_23, getitem_393, 1024);
    permute_47 = floordiv_23 = getitem_393 = None
    transformer_h_23_self_attention_dense = getattr(self.transformer.h, "23").self_attention.dense(reshape_96);
    reshape_96 = None
    dropout_46 = torch.nn.functional.dropout(transformer_h_23_self_attention_dense, p=0.0, training=False,
                                             inplace=False);
    transformer_h_23_self_attention_dense = None
    add_95 = add_94 + dropout_46;
    add_94 = dropout_46 = None
    transformer_h_23_post_attention_layernorm = getattr(self.transformer.h, "23").post_attention_layernorm(add_95)
    transformer_h_23_mlp_dense_h_to_4h = getattr(self.transformer.h, "23").mlp.dense_h_to_4h(
        transformer_h_23_post_attention_layernorm);
    transformer_h_23_post_attention_layernorm = None
    mul_237 = transformer_h_23_mlp_dense_h_to_4h * 0.5
    mul_238 = 0.79788456 * transformer_h_23_mlp_dense_h_to_4h
    mul_239 = 0.044715 * transformer_h_23_mlp_dense_h_to_4h
    mul_240 = mul_239 * transformer_h_23_mlp_dense_h_to_4h;
    mul_239 = transformer_h_23_mlp_dense_h_to_4h = None
    add_96 = 1 + mul_240;
    mul_240 = None
    mul_241 = mul_238 * add_96;
    mul_238 = add_96 = None
    tanh_23 = torch.tanh(mul_241);
    mul_241 = None
    add_97 = 1.0 + tanh_23;
    tanh_23 = None
    mul_242 = mul_237 * add_97;
    mul_237 = add_97 = None
    transformer_h_23_mlp_dense_4h_to_h = getattr(self.transformer.h, "23").mlp.dense_4h_to_h(mul_242);
    mul_242 = None
    dropout_47 = torch.nn.functional.dropout(transformer_h_23_mlp_dense_4h_to_h, p=0.0, training=False, inplace=False);
    transformer_h_23_mlp_dense_4h_to_h = None
    add_98 = add_95 + dropout_47;
    add_95 = dropout_47 = None
    transformer_ln_f = self.transformer.ln_f(add_98);
    add_98 = None
    lm_head = self.lm_head(transformer_ln_f);
    transformer_ln_f = None
    return {'logits': lm_head}

# To see more debug info, please use `graph_module.print_readable()`