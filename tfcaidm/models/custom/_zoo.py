"""Defined model blocks"""

import tfcaidm.models.layers.learner as learner
import tfcaidm.models.layers.transform as transform

import tfcaidm.models.nn.ae as ae
import tfcaidm.models.nn.unet as unet
import tfcaidm.models.nn.unetpp as unetpp
import tfcaidm.models.nn.unet3p as unet3p

import tfcaidm.models.blocks.atrous_block as atrous_block
import tfcaidm.models.blocks.attention_gate as attention_gate
import tfcaidm.models.blocks.cbam_block as cbam_block
import tfcaidm.models.blocks.csp_block as csp_block
import tfcaidm.models.blocks.convgru_cell as convgru_cell
import tfcaidm.models.blocks.dense_block as dense_block
import tfcaidm.models.blocks.eca_block as eca_block
import tfcaidm.models.blocks.inception_block as inception_block
import tfcaidm.models.blocks.psp_block as psp_block
import tfcaidm.models.blocks.se_block as se_block
import tfcaidm.models.blocks.separable_conv as separable_conv
import tfcaidm.models.blocks.u2net_block as u2net_block

import tfcaidm.models.utils.head as head
import tfcaidm.models.utils.tasks as tasks

conv_types = {
    "conv": learner.conv,
    "small": separable_conv.separable_conv,
    "smaller": separable_conv.smaller_separable_conv,
    "smallest": separable_conv.smallest_separable_conv,
    "channel_spatial": separable_conv.channel_spatial_conv,
}

tran_types = {
    "conv": learner.tran,
    "small": separable_conv.separable_tran,
    "smaller": separable_conv.smaller_separable_tran,
    "smallest": separable_conv.smallest_separable_tran,
    "channel_spatial": separable_conv.channel_spatial_tran,
}

pool_types = {
    "conv": learner.conv,
    "max": transform.max_pool,
    "avg": transform.average_pool,
    "aspp": atrous_block.aspp,
    "acsp": atrous_block.acsp,
    "wasp": atrous_block.wasp,
}

eblocks = {
    "conv": learner.conv,
    "aspp": atrous_block.aspp,
    "acsp": atrous_block.acsp,
    "wasp": atrous_block.wasp,
    "cbam": cbam_block.cbam,
    "csp": csp_block.csp,
    "dense": dense_block.dense,
    "eca": eca_block.eca,
    "inception": inception_block.inception,
    "psp": psp_block.psp,
    "se": se_block.se,
    "u2net": u2net_block.u2net,
}

dblocks = {
    "conv": learner.tran,
    "convgru": convgru_cell.convgru,
    "attention": attention_gate.attention_gate,
}

heads = {
    "encoder_classifier": head.Encoder.last_layer,
    "encoder_multi_scale_classifier": head.Encoder.multi_scale,
    "decoder_classifier": head.Decoder.last_layer,
    "decoder_multi_scale_classifier": head.Decoder.multi_scale,
    "decoder_deep_supervision_classifier": head.Decoder.deep_supervision,
    "decoder_complex_supervision_classifier": head.Decoder.complex_supervision,
}

task_types = {
    "auto": tasks.auto_task_selector,
}

models = {
    "ae": ae.ae,
    "unet": unet.unet,
    "unet++": unetpp.unetpp,
    "unet3+": unet3p.unet3p,
}
