#pragma once

#include <Eigen/Eigen>
#include <Eigen/SVD>

#include <Eigen/LU> 
#include <Eigen/QR>

//////////////////// pseudo inverse //////////////////// 

// WORKING 
template<typename T = Eigen::MatrixXd > T pinv(const T& M)
{
	// TODO
	// \warning: Do not compute \c this->pseudoInverse()*rhs to solve a linear systems.
	//	* It is more efficient and numerically stable to call \c this->solve(rhs).
	return M.completeOrthogonalDecomposition().pseudoInverse();
}

// WORKING 
template<typename T = Eigen::MatrixXd> T pseudoInverse(const T &a, double epsilon = std::numeric_limits<double>::epsilon())
{
	// code from http://eigen.tuxfamily.org/bz/show_bug.cgi?id=257
	Eigen::JacobiSVD< T > svd(a, Eigen::ComputeThinU | Eigen::ComputeThinV);
	double tolerance = epsilon * std::max(a.cols(), a.rows()) *svd.singularValues().array().abs()(0);
	return svd.matrixV() *  (svd.singularValues().array().abs() > tolerance).select(svd.singularValues().array().inverse(), 0).matrix().asDiagonal() * svd.matrixU().adjoint();
}

//////////////////// linear regression //////////////////// 

// given Y = W*X
// solve the equation system for W, minimizing squared error
// example: getting weight matrix of a linear perceptron
// with X = some training data (each column is one dataset)
// and Y  = class labels, one-hot-coded 
// x = 17 24  1    y = 1 0 0
//     23  5  7        0 1 0
//      4  6 13        0 0 1
//     10 12 19
//     11 18 25
// W = linear_regression_column_major(x,y);
// yy = W*x now should be very close to true class labels y
// please note that data in X and Y must be column major similiar to matlab style programming 
template<typename T = Eigen::MatrixXd > T linear_regression(const T& X, const T& Y)
{
	//	It is more efficient and numerically stable to call \c this->solve(rhs).
	return X.transpose().completeOrthogonalDecomposition().solve(Y.transpose()).transpose();
}

// BROKEN (?)

//-----------------------------------------------------------------------------
//! \brief pseudo_inverse
//! \brief Computes the pseudo-Inverse of a rectangular Matrix
//! \return true if the the SVD succeed
//! \param[in] The Original Matrix
//! \param[in] The Pseudo-Inverted Matrix
//! \tparam The type contained by the matrix (double, int...)
template<typename Scalar>
bool pinv_broken(const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &a, Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &a_pinv)
{
    // see : http://en.wikipedia.org/wiki/Moore-Penrose_pseudoinverse#The_general_case_and_the_SVD_method

    if ( a.rows()<a.cols() )
        return false;

    // JacobiSVD - fast and accurate for small matrixes
	// Eigen::JacobiSVD< Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> > svdA(a);

	// BDCSVD - remains fast for large matrixes
	Eigen::BDCSVD< Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> > svdA(a);

    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> vSingular = svdA.singularValues();

    // Build a diagonal matrix with the Inverted Singular values
    // The pseudo inverted singular matrix is easy to compute :
    // is formed by replacing every nonzero entry by its reciprocal (inversing).
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> vPseudoInvertedSingular(svdA.matrixV().cols(),1);
        
    for (int iRow =0; iRow<vSingular.rows(); iRow++)
    {
        if ( fabs(vSingular(iRow))<=1e-10 ) // Todo : Put epsilon in parameter
        {
            vPseudoInvertedSingular(iRow,0)=0.;
        }
        else
        {
            vPseudoInvertedSingular(iRow,0)=1./vSingular(iRow);
        }
    }

    // A little optimization here 
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> mAdjointU = svdA.matrixU().adjoint().block(0,0,vSingular.rows(),svdA.matrixU().adjoint().cols());

    // Pseudo-Inversion : V * S * U'
    a_pinv = (svdA.matrixV() *  vPseudoInvertedSingular.asDiagonal()) * mAdjointU  ;

    return true;
}
//-----------------------------------------------------------------------------


#ifdef __TEST_THIS_MODULE__

void test_module()
{
	using namespace Eigen;
	using namespace std;
	MatrixXd M = MatrixXd::Zero(5, 5);
	// magic numbers - matlab: magic(5)
	
	M << 17, 24,  1, 8,  15,
		 23,  5,  7, 14, 16,
		  4,  6, 13, 20, 22,
		 10, 12, 19, 21,  3,
		 11, 18, 25,  2,  9;
	
	cout << "M=\n" << M << "\n\n";

	auto M_inv = pinv(M);

	cout << "pinv(M) =\n" << M_inv << "\n\n";

	cout << "should be:\n" <<
    "-0.0049    0.0512   -0.0354    0.0012    0.0034\n"
    " 0.0431   -0.0373   -0.0046    0.0127    0.0015\n"
    "-0.0303    0.0031    0.0031    0.0031    0.0364\n"
    " 0.0047   -0.0065    0.0108    0.0435   -0.0370\n"
    " 0.0028    0.0050    0.0415   -0.0450    0.0111\n"
		<< "\n\n";

	auto tmp = M * M_inv;
	cout << "M*pinv(M) =\n" << tmp << "\n\n";

	cout << "sum(M*pinv(M)) - 5.0 = " << tmp.sum() - 5.0f;

	cout << "//////// linear regression by equation solving //////\n\n";
	// y = W * x
	// what is W ?
	MatrixXd x = M.block(0,0,5,3);
	MatrixXd y = MatrixXd::Identity(3, 3);
	cout << "x=\n" << x << "\n\n";
	cout << "y=\n" << y << "\n\n";
	MatrixXd W = linear_regression(x, y);
	cout << "W=\n" << W << "\n\n";

	// check if equation indeed gets (almost) solved
	cout << "W*x=\n" << W*x << "\n\n";
	cout << "sum(W*x) - 3.0 = " << (W * x).sum() - 3.0f;

	// compare with pinv:
	W = y * pinv(x);
	cout << "W=\n" << W << "\n\n";
	cout << "W*x=\n" << W * x << "\n\n";
	cout << "sum(W*x) - 3.0 = " << (W * x).sum() - 3.0f << "\n\n";;

}

#endif