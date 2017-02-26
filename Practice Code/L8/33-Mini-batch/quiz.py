def batches(batch_size,features,labels):
    assert len(features) == len(labels)
    output_batches = []
    for start_point in range(0,len(features),batch_size):
        end_point = start_point + batch_size
        batch = [features[start_point:end_point], labels[start_point:end_point]]
        output_batches.append(batch)

    return output_batches
