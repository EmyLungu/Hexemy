import subprocess

fileName = "samples/t1"
func_name = "func"

OPT = ["O0", "O1", "O2", "O3"]
for opt_state in OPT:
    output_file = fileName + "_" + opt_state
    input_file = fileName + ".c"
    compile_command = f"gcc -o {output_file}.o {input_file} -{opt_state} -lm"
    subprocess.run(compile_command, shell=True, check=True)
    compile_command = f"objdump -d {output_file}.o > {output_file}.s"
    subprocess.run(compile_command, shell=True, check=True)

    input_asm = ""
    with open(output_file + ".s") as f:
        asm = f.read()
        if "<" + func_name + ">:" not in asm:
            raise ValueError("compile fails")
        asm = (
            "<"
            + func_name
            + ">:"
            + asm.split("<" + func_name + ">:")[-1].split("\n\n")[0]
        )
        asm_clean = ""
        asm_sp = asm.split("\n")
        for tmp in asm_sp:
            if len(tmp.split("\t")) < 3 and "00" in tmp:
                continue
            idx = min(len(tmp.split("\t")) - 1, 2)
            tmp_asm = "\t".join(tmp.split("\t")[idx:])
            tmp_asm = tmp_asm.split("#")[0].strip()
            asm_clean += tmp_asm + "\n"
    input_asm = asm_clean.strip()
    input_asm_prompt = input_asm.strip()
    with open(fileName + "_" + opt_state + ".asm", "w", encoding="utf-8") as f:
        f.write(input_asm_prompt)
