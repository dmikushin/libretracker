#ifndef __EIGEN_HELPERS__
#define __EIGEN_HELPERS__


	template<class T> bool load_matrix(Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& m, const string& fname)
	{
		fstream f;
		f.open(fname.c_str(), ios::in);
		if(f.is_open())
			return load_matrix(m, f);
		else
			return false;
		return true;
	}

	template<class T> bool load_matrix(Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& m, istream& is)
	{
		T x;
		
		for (int r=0; r<m.rows();r++)
		{
			for (int c=0;c<m.cols();c++)	
			{
				if(is.eof()) return false;
				is >> x; 
				m(r, c) = x;
			}
		}
		return true;
	}

#endif