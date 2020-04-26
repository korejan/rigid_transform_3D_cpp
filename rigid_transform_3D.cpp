#include <cstdlib>
#include <ctime>
#include <iostream>
#include <Eigen/Geometry>

//
// Ported/Adapted from: https://github.com/nghiaho12/rigid_transform_3D/blob/master/rigid_transform_3D.py
// Based on article: http://nghiaho.com/?page_id=671
//
using PointSet = Eigen::Matrix<float, 3, Eigen::Dynamic>;
auto rigid_transform_3D(const PointSet& A, const PointSet& B) -> std::tuple<Eigen::Matrix3f, Eigen::Vector3f>
{
	static_assert(PointSet::RowsAtCompileTime == 3);
	assert(A.cols() == B.cols());

	// find mean column wise
	const Eigen::Vector3f centroid_A = A.rowwise().mean();
	const Eigen::Vector3f centroid_B = B.rowwise().mean();

	// subtract mean
	PointSet Am = A.colwise() - centroid_A;
	PointSet Bm = B.colwise() - centroid_B;

	PointSet H = Am * Bm.transpose();

	//
	//# sanity check
	//#if linalg.matrix_rank(H) < 3:
	//	#    raise ValueError("rank of H = {}, expecting 3".format(linalg.matrix_rank(H)))
	//

	// find rotation
	Eigen::JacobiSVD<Eigen::Matrix3Xf> svd = H.jacobiSvd(Eigen::DecompositionOptions::ComputeFullU | Eigen::DecompositionOptions::ComputeFullV);
	const Eigen::Matrix3f& U = svd.matrixU();
	Eigen::MatrixXf V = svd.matrixV();
	Eigen::Matrix3f R = V * U.transpose();

	// special reflection case
	if (R.determinant() < 0.0f)
	{
		V.col(2) *= -1.0f;
		R = V * U.transpose();
	}

	const Eigen::Vector3f t = -R * centroid_A + centroid_B;

	return std::make_tuple(R, t);
}

int main()
{
	// Test with random data
	std::srand(static_cast<unsigned int>(std::time(0)));

	// Random rotationand translation
	Eigen::Matrix3f R = Eigen::Matrix3f::Random();
	Eigen::Vector3f t = Eigen::Vector3f::Random();

	// make R a proper rotation matrix, force orthonormal
	Eigen::JacobiSVD<Eigen::Matrix3f> svd = R.jacobiSvd(Eigen::DecompositionOptions::ComputeFullU | Eigen::DecompositionOptions::ComputeFullV);
	const Eigen::Matrix3f U = svd.matrixU();
	Eigen::Matrix3f Vt = svd.matrixV();
	R = U * Vt;
	
	// remove reflection
	if (R.determinant() < 0.0f)
	{
		Vt.col(2) *= -1.0f;
		R = U * Vt;
	}

	// number of points
	constexpr const std::size_t N = 10;

	PointSet A = PointSet::Random(3, N);
	PointSet B = (R * A).colwise() + t;

	// Recover R and t
	const auto [ret_R, ret_t] = rigid_transform_3D(A, B);

	// Compare the recovered R and t with the original
	PointSet B2 = (ret_R * A).colwise() + ret_t;

	// Find the root mean squared error
	PointSet err = B2 - B;
	err = err.cwiseProduct(err);
	const float rmse = std::sqrt(err.sum() / static_cast<float>(N));

	std::cout << "Points A\n" << A << '\n';

	std::cout << "Points B\n" << B << '\n';

	std::cout << "Ground truth rotation\n" << R << '\n';

	std::cout << "Recovered rotation\n" << ret_R << '\n';

	std::cout << "Ground truth translation\n" << t << '\n';

	std::cout << "Recovered translation\n" << ret_t << '\n';

	std::cout << "RMSE: " << rmse << '\n';

	if (rmse < 1e-5f)
		std::cout << "Everything looks good!\n";
	else
		std::cout << "Hmm something doesn't look right ...\n";
}
