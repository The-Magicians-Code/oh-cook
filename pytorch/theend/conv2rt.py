from subprocess import call
from glob import glob
import argparse

onnx = [i[:-5] for i in glob("models/*.onnx")]
engine = [i[:-7] for i in glob("models/*.engine")]

print(onnx)
print(engine)

not_converted = [i for i in onnx if not i in engine]
print(not_converted)
for model in not_converted:
    call(f"trtexec --onnx={model}.onnx --saveEngine={model}.engine --inputIOFormats=fp16:chw --outputIOFormats=fp16:chw --fp16 --noTF32 --workspace={1 << 30}".split()) # --verbose optional