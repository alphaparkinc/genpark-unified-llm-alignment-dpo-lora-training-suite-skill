from client import UnifiedLlmAlignmentDpoLoraTrainingSuiteClient

def main():
    client = UnifiedLlmAlignmentDpoLoraTrainingSuiteClient()
    res = client.launch_alignment_training_run('ultrafeedback_binarized', 'ORPO_ALIGNMENT', 2)
    print('Alignment Run: ' + res['training_run_id'] + ' (Algorithm: ' + res['algorithm'] + ')')
    print('Dataset: ' + res['dataset'] + ' | Epochs: ' + str(res['epochs']))
    print('Reward Margin Gain: +' + str(res['eval_reward_margin_improvement_pct']) + '% | Loss Converged: ' + str(res['dpo_loss_converged']))
    print('Adapter Weights: ' + res['exported_adapter_weights_url'])

if __name__ == '__main__':
    main()
