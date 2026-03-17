# 璁粌鏂规鎬昏锛圓utoDL + 7B锛?
## 1. 鍩烘ā閫夋嫨

### 鏂囨湰妯″瀷

- `Qwen/Qwen2.5-7B-Instruct`

### 瑙嗚妯″瀷

- `FreedomIntelligence/HuatuoGPT-Vision-7B-Qwen2.5VL`

閫夋嫨鍘熷洜锛?
- 涓枃鑳藉姏杈冨己
- 鎸囦护璺熼殢鍜屽伐鍏疯皟鐢ㄨ〃鐜扮ǔ
- QLoRA 鐢熸€佹垚鐔?- 姣旇緝閫傚悎浣犺繖涓€滃尰鐤?Agent + RAG + 澶氳疆闂瘖鈥濈殑椤圭洰

## 2. AutoDL 鏁版嵁鐩樺竷灞€

缁熶竴浣跨敤锛?
- `/root/autodl-tmp/medagent`

寤鸿瀛愮洰褰曪細

- `models/`锛氬熀妯?- `datasets/sft/`锛歋FT 鏁版嵁
- `datasets/rl/`锛欰gentic-RL 鏁版嵁
- `datasets/rag_raw/`锛歊AG 鍘熷鏂囨。
- `rag/`锛氬垏鍒嗙粨鏋滃拰绱㈠紩
- `outputs/adapters/`锛歀oRA adapter
- `wandb/`锛歐&B 鏃ュ織
- `hf_cache/`锛欻ugging Face 缂撳瓨

鍒濆鍖栧懡浠わ細

```bash
bash scripts/autodl_prepare_workspace.sh /root/autodl-tmp/medagent
```

## 3. SFT 鏁版嵁閫夋嫨

鎺ㄨ崘涓绘枡锛?
- `BillGPT/Chinese-medical-dialogue-data`
- `wangrongsheng/cMedQA-V2.0`
- `FreedomIntelligence/Medical-R1-Distill-Data-Chinese`
- `FreedomIntelligence/HuatuoGPT-sft-data-v1`
- `shibing624/medical` 涓繃婊ゅ悗鐨勯珮璐ㄩ噺涓枃瀛愰泦

杩樿琛ュ唴閮ㄦ暟鎹細

- 棣栫▼闂瘖杞ㄨ抗
- 绉戞櫘瑙ｉ噴杞ㄨ抗
- 鍒嗚瘖缁撹
- 鎶ュ憡瑙ｉ噴
- 闂嵂鍜岃嵂鐗╃蹇?- 缁撳悎闀挎湡璁板繂鐨勫璇婂満鏅?
鏈€缁堢粺涓€鎴愶細

```json
{"input":"...", "output":"..."}
```

鍏朵腑棣栫▼闂瘖杩欎竴妗惰鐗瑰埆娉ㄦ剰锛?
- 涓嶈鐩存帴鐢ㄩ暱绡囪В閲婂瀷绛旀鍘昏 `BianQue-Intake`
- 鏇撮€傚悎璁垚鈥滀竴闂竴绛斺€濈殑杩介棶椋庢牸
- 闀胯В閲婃牱鏈暀缁欏悗缁?`LiShiZhen-Education` / 鎶ュ憡瑙ｈ adapter

## 4. QLoRA 闃舵

鍏堝仛 SFT adapter 璁粌锛?
```bash
export WANDB_API_KEY=xxx
bash scripts/run_sft_with_wandb.sh /root/autodl-tmp/medagent
```

鎴栫洿鎺ヨ繍琛岋細

```bash
python3 scripts/train_qlora.py \
  --base-model /root/autodl-tmp/medagent/models/qwen2.5-7b-instruct \
  --train-file /root/autodl-tmp/medagent/datasets/sft/train_v1.jsonl \
  --replay-file /root/autodl-tmp/medagent/datasets/sft/replay_v0.jsonl \
  --task general_intake \
  --dataset-name med_sft_v1 \
  --output-dir /root/autodl-tmp/medagent/outputs/adapters/general_intake_v1 \
  --adapter-bank-dir /root/autodl-tmp/medagent/outputs/adapters \
  --cache-dir /root/autodl-tmp/medagent/hf_cache \
  --wandb-project medagent-7b \
  --wandb-run-name sft-general-intake-v1 \
  --wandb-dir /root/autodl-tmp/medagent/wandb
```

## 5. Agentic-RL 鏁版嵁绛栫暐

鍏紑鍙洿鎺ョ敤鐨勬暟鎹彧閫傚悎浣滅瀛愶紝涓嶈冻浠ヨ鐩栫湡瀹為棶璇婅涓恒€?
鏇村悎鐞嗙殑缁勬垚鏄細

- 灏忚妯″叕寮€ medical preference / reward 鏁版嵁
- 浣犵殑澶?Agent 绯荤粺鐪熷疄鏃ュ織
- 鍩轰簬 benchmark 鍜岀梾渚嬪寘鍚堟垚鐨?agentic preference pairs

鎺ㄨ崘绗竴鐗堥厤姣旓細

- `60%` 鍚堟垚 agentic preference
- `20%` 澶?Agent 鐪熷疄鏃ュ織鍥炴祦
- `10%` 鍏紑 medical preference 鏁版嵁
- `10%` 楂橀闄╀汉宸ユ瀯閫犺礋鏍锋湰

閲嶇偣 failure mode锛?
- 婕忛棶杩囨晱鍙?- 婕忛棶鎱㈢梾鍜岄暱鏈熺敤鑽?- 鐢锋€ц闂€€瀛?- 鑲查緞濂虫€ц闂濞犲嵈娌￠棶
- 鍗遍櫓淇″彿娌℃湁鍗囩骇鎬ヨ瘖
- 鏄庢槑闇€瑕佹姤鍛?鑽墿宸ュ叿鍗翠笉璋冪敤

## 6. Agentic-RL 璁粌闃舵

绗竴姝ュ缓璁厛鍋?DPO/ORPO 涓€绫荤殑鍋忓ソ浼樺寲锛岃€屼笉鏄竴涓婃潵灏卞仛澶嶆潅鍦ㄧ嚎 RL銆?
绀轰緥锛?
```bash
python3 scripts/train_agentic_rl.py \
  --base-model /root/autodl-tmp/medagent/models/qwen2.5-7b-instruct \
  --pairs-file /root/autodl-tmp/medagent/datasets/rl/agentic_pairs_v1.jsonl \
  --task agentic_policy \
  --output-dir /root/autodl-tmp/medagent/outputs/adapters/agentic_policy_v1 \
  --adapter-bank-dir /root/autodl-tmp/medagent/outputs/adapters \
  --cache-dir /root/autodl-tmp/medagent/hf_cache \
  --wandb-project medagent-7b \
  --wandb-run-name rl-agentic-policy-v1 \
  --wandb-dir /root/autodl-tmp/medagent/wandb
```

## 7. 澶氭ā鎬侀樁娈?
瑙嗚鍒嗘敮閲嶇偣鍏堝仛锛?
- 妫€楠屾姤鍛?- 鍖栭獙鍗?- 缁撴瀯鍖栧浘鏂囬棶绛?
涓嶈涓€涓婃潵灏卞仛閲嶅奖鍍忚瘖鏂€?
璁粌鏁版嵁寤鸿鍏堣仛鐒︼細

- 鎶ュ憡鍥剧墖 + 闂
- OCR 鏂囨湰 + 闂
- 杈撳嚭寮傚父鎽樿銆侀闄╂彁绀哄拰涓嬩竴姝ュ缓璁?
## 8. 鏈€缁堣瘎娴?
璺戠绾?benchmark锛?
```bash
python3 -m medagent.benchmark.run --dataset data/benchmark_cases.json
```

鍐嶈窇澶氳疆鐥呬汉妯℃嫙 benchmark锛?
```bash
python scripts/run_patient_sim_benchmark.py --scenarios data/sim_patient_cases.json
```

## 9. W&B 寤鸿璺熻釜鎸囨爣

- `loss`
- `learning_rate`
- `grad_norm`
- 楂橀闄╃棁鐘跺彫鍥炵巼
- grounding 寮曠敤鐜?- 宸ュ叿璋冪敤鎴愬姛鐜?- benchmark 鎬诲垎

## 10. 寤鸿鍜屽摢浜涙枃妗ｈ仈鍔ㄧ湅

- `../system/architecture.md`
- `./agentic_rl_synthesis_zh.md`
- `./agentic_rl_dataset_survey_zh.md`
- `../system/clinical_first_visit_prompt_zh.md`
- `../evaluation/benchmark_plan.md`

## 琛ュ厖鏂囨。

- [Huatuo 绯诲垪妯″瀷璋冪爺](./huatuo_series_research_zh.md)
- [鍩烘ā閫夋嫨鍐崇瓥](./base_model_choice_zh.md)
- [绗竴闃舵璁粌鏂规](./stage1_training_plan_zh.md)
- [RAG 鏂囨。婧愭柟妗圿(./rag_sources_zh.md)

