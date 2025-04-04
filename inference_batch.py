import pandas as pd
from omegaconf import OmegaConf

from src.dataloaders.spades import create_dataset
from src.models.mobilenet import MobilenetModel
from src.utils.metrics import compute_pose_score
import tensorflow_graphics.geometry.transformation as tfgt

if __name__ == '__main__':

    config_path = "configs/mobilenet.yaml"
    try:
        config = OmegaConf.load(config_path)
    except Exception as e:
        print("Error loading YAML:", e)

    # dataset creation
    test_df = pd.read_csv(config.root.test_data)
    test_data = {col: test_df[col].values for col in test_df.columns}

    test_dataset = create_dataset(test_data,
                                  batch_size=config.training.batch_size,
                                  input_size=config.data.input_size[:2],
                                  is_training=False,
                                  cache_dir=None)

    # tf.keras.utils.get_custom_objects().update({'PoseEstimationLoss': PoseEstimationLoss})
    input_image_size = config.data.input_size
    input_shape = list(config.data.input_size)  # Convert ListConfig to a standard list
    model = MobilenetModel(input_size=input_shape, pretrained=config.model.pretrained)
    model.build(input_shape=(None, *input_shape))  # None is for batch size
    model.load_weights(config.model.trained_weights)
    predictions = model.predict(test_dataset, verbose=1)
    # print(predictions[0])

    pos_preds = predictions[0]  # Shape: (num_samples, 3)
    quat_preds = tfgt.quaternion.normalize(predictions[1])

    # This regex will remove the full prefix including the RT-folder (like RT001/, RT023/, etc.)
    test_df['filepath'] = test_df['filepath'].str.replace(r'^/home/arunkumar/datasets/SPADES/synthetic/lnes/RT\d+/', '',
                                                          regex=True)
    filepaths = test_df['filepath'].values
    print(len(filepaths), len(pos_preds), len(quat_preds))

    df = pd.DataFrame({
        'filepath': filepaths,
        'Tx': pos_preds[:, 0],
        'Ty': pos_preds[:, 1],
        'Tz': pos_preds[:, 2],
        'Qx': quat_preds[:, 0],
        'Qy': quat_preds[:, 1],
        'Qz': quat_preds[:, 2],
        'Qw': quat_preds[:, 3],
    })

    pred_csv = "./tmp/test_predictions.csv"
    gt_csv = "./tmp/test_ground_truth.csv"
    df.to_csv(pred_csv, index=False)

    test_df.to_csv(gt_csv, index=False)


    pos_score, ori_score, pose_score = compute_pose_score(gt_csv, pred_csv)


