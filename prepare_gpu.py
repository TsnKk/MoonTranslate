"""Build an equivalent DirectML graph: flatten high-rank MatMul batches.
Weights, image resolution and inpainting masks are unchanged.
Requires the optional onnx package; used only at setup time.
"""
import os
from pathlib import Path
import onnx
from onnx import helper as h, numpy_helper
import numpy as np
from prepare_engine import ROOT as root
from prepare_inpainting import install
install()
m=onnx.load(str(root/'inpainting/lama-manga.onnx'))
shaped=onnx.shape_inference.infer_shapes(m)
ranks={v.name:len(v.type.tensor_type.shape.dim) for v in shaped.graph.value_info}
nodes=[];count=0
for node in m.graph.node:
    if node.op_type!='MatMul' or max(ranks.get(v,0) for v in node.input)<=4:
        nodes.append(node);continue
    a,b=node.input;ra,rb=ranks[a],ranks[b]
    assert (ra==2 and rb>4) or (rb==2 and ra>4),(node.name,ra,rb)
    big=b if rb>4 else a;other=a if rb>4 else b
    prefix=f'dml_flatten_{count}'
    def constant(suffix,values):
        name=prefix+suffix
        nodes.append(h.make_node('Constant',[],[name],value=numpy_helper.from_array(np.array(values,np.int64))))
        return name
    last2=constant('_last2',[-2,-1]);minus1=constant('_minus1',[-1]);batchindices=constant('_batch',list(range(max(ra,rb)-2)))
    one=constant('_one',[-1]);two=constant('_two',[-2])
    nodes.append(h.make_node('Shape',[big],[prefix+'_shape']))
    nodes.append(h.make_node('Shape',[other],[prefix+'_other_shape']))
    nodes.append(h.make_node('Gather',[prefix+'_shape',last2],[prefix+'_matrix_shape'],axis=0))
    nodes.append(h.make_node('Concat',[minus1,prefix+'_matrix_shape'],[prefix+'_flat_shape'],axis=0))
    nodes.append(h.make_node('Reshape',[big,prefix+'_flat_shape'],[prefix+'_flat']))
    nodes.append(h.make_node('MatMul',[a if rb>4 else prefix+'_flat',prefix+'_flat' if rb>4 else b],[prefix+'_output'],name=node.name))
    nodes.append(h.make_node('Gather',[prefix+'_shape',batchindices],[prefix+'_batch_shape'],axis=0))
    nodes.append(h.make_node('Gather',[prefix+'_other_shape' if rb>4 else prefix+'_shape',two],[prefix+'_m'],axis=0))
    nodes.append(h.make_node('Gather',[prefix+'_shape' if rb>4 else prefix+'_other_shape',one],[prefix+'_n'],axis=0))
    nodes.append(h.make_node('Concat',[prefix+'_batch_shape',prefix+'_m',prefix+'_n'],[prefix+'_restore_shape'],axis=0))
    nodes.append(h.make_node('Reshape',[prefix+'_output',prefix+'_restore_shape'],list(node.output)))
    count+=1
del m.graph.node[:];m.graph.node.extend(nodes)
onnx.checker.check_model(m)
target=root/'inpainting/lama-manga-dml-v1.onnx'
onnx.save(m,str(target));print('Flattened',count,'MatMul nodes:',target)
