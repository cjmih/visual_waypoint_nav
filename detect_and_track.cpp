
cv::Mat accumulate_rotation(const cv::Mat& prev_cumulative_R, const cv::Mat& current_R) {
    return current_R * prev_cumulative_R;
}

void decomposeHomographyMat(const Mat& H, const Mat& K, vector<Mat>& rotations, vector<Mat>& translations, vector<Mat>& normals) {
    cv::decomposeHomographyMat(H, K, rotations, translations, normals);
    
    cout << "Possible solutions:" << endl;
    for (size_t i = 0; i < rotations.size(); i++) {
        cout << "Solution " << i + 1 << ":" << endl;
        cout << "Rotation:" << endl << rotations[i] << endl;
        cout << "Translation:" << endl << translations[i] << endl;
        cout << "Normal:" << endl << normals[i] << endl;
    }
}
