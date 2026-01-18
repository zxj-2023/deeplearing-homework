import json
import os
from src.utils import normalize_answer
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import logging
logger = logging.getLogger(__name__)

class Evaluator:
    def __init__(self, llm_model, predictions_path):
        # llm_model: 用于判定答案是否正确的 LLM 接口封装。
        # predictions_path: 预测结果 JSON 路径（包含 pred_answer 与 gold_answer）。
        self.llm_model = llm_model
        self.predictions_path = predictions_path
        self.prediction_results = self.load_predictions()

    def load_predictions(self):
        # 读取预测结果列表，每个元素是一条样本的预测信息。
        with open(self.predictions_path, "r", encoding="utf-8") as f:
            prediction_results = json.load(f)
        return prediction_results
    
    def calculate_llm_accuracy(self,pre_answer,gold_ans):
        # 通过 LLM 判定生成答案是否与标准答案一致。
        system_prompt = """You are an expert evaluator. 
        """
        user_prompt = f"""Please evaluate if the generated answer is correct by comparing it with the gold answer.
        Generated answer: {pre_answer}
        Gold answer: {gold_ans}

        The generated answer should be considered correct if it:
        1. Contains the key information from the gold answer
        2. Is factually accurate and consistent with the gold answer
        3. Does not contain any contradicting information

        Respond with ONLY 'correct' or 'incorrect'.
        Response:
        """
        response = self.llm_model.infer([{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}])
        if response.strip().lower() == "correct":
            return 1.0
        else:
            return 0.0

    def calculate_contain(self,pre_answers,gold_ans):
        # 基于字符串包含关系的简单匹配指标（归一化后判断 gold 是否被包含）。
        if pre_answers is None or pre_answers == "" or (isinstance(pre_answers, str) and pre_answers.strip() == ""):
            return 0            
        if gold_ans is None or gold_ans == "" or (isinstance(gold_ans, str) and gold_ans.strip() == ""):
            return 0
        s1 = normalize_answer(pre_answers)
        s2 = normalize_answer(gold_ans)
        if s2 in s1:
            return 1
        else:
            return 0
            
    def evaluate_sig_sample(self,idx,prediction):
        # 评估单条样本，返回索引与两种指标的得分。
        pre_answer = prediction["pred_answer"]
        gold_ans = prediction["gold_answer"]
        # llm_acc = 0.0
        llm_acc = self.calculate_llm_accuracy(pre_answer, gold_ans)
        contain_acc = self.calculate_contain(pre_answer, gold_ans)
        return idx, llm_acc, contain_acc

    def evaluate(self,max_workers):
        # 多线程评估所有样本，并在结束后写回结果与汇总指标。
        llm_scores = [0.0] * len(self.prediction_results)
        contain_scores = [0.0] * len(self.prediction_results)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self.evaluate_sig_sample, idx, pred): idx 
                for idx, pred in enumerate(self.prediction_results)
            }

            completed = 0
            total_llm_score = 0.0
            total_contain_score = 0.0
            # 进度条实时展示当前平均指标。
            pbar = tqdm(total=len(futures), desc="Evaluating samples", unit="sample")
            for future in as_completed(futures):
                idx, llm_acc, contain_acc  = future.result()
                llm_scores[idx] = llm_acc
                contain_scores[idx] = contain_acc
                self.prediction_results[idx]["llm_accuracy"] = llm_acc
                self.prediction_results[idx]["contain_accuracy"] = contain_acc
                total_llm_score += llm_acc
                total_contain_score += contain_acc
                completed += 1
                current_llm_acc = total_llm_score / completed
                current_contain_acc = total_contain_score / completed
                pbar.set_postfix({
                    'LLM_Acc': f'{current_llm_acc:.3f}',
                    'Contain_Acc': f'{current_contain_acc:.3f}'
                })
                pbar.update(1)
            pbar.close()

        # 计算总体平均指标。
        llm_accuracy = sum(llm_scores) / len(llm_scores)
        contain_accuracy = sum(contain_scores) / len(contain_scores)

        # 记录日志并落盘保存评估结果。
        logger.info(f"Evaluation Results:")
        logger.info(f"  LLM Accuracy: {llm_accuracy:.4f} ({sum(llm_scores)}/{len(llm_scores)})")
        logger.info(f"  Contain Accuracy: {contain_accuracy:.4f} ({sum(contain_scores)}/{len(contain_scores)})")
        with open(self.predictions_path, "w", encoding="utf-8") as f:
            json.dump(self.prediction_results, f, ensure_ascii=False, indent=4)
        
        # 额外输出汇总指标，便于快速查看。
        with open(os.path.join(os.path.dirname(self.predictions_path), "evaluation_results.json"), "w", encoding="utf-8") as f:
            json.dump({"llm_accuracy": llm_accuracy, "contain_accuracy": contain_accuracy}, f, ensure_ascii=False, indent=4)
        return llm_accuracy, contain_accuracy
