#include "RecoLocalTracker/SiPixelRecHits/interface/PixelCPEGenericBase.h"

PixelCPEGenericBase::PixelCPEGenericBase(edm::ParameterSet const& conf,
                                         const MagneticField* mag,
                                         const TrackerGeometry& geom,
                                         const TrackerTopology& ttopo,
                                         const SiPixelLorentzAngle* lorentzAngle,
                                         const SiPixelGenErrorDBObject* genErrorDBObject,
                                         const SiPixelLorentzAngle* lorentzAngleWidth = nullptr)
    : PixelCPEBase(conf, mag, geom, ttopo, lorentzAngle, genErrorDBObject, nullptr, lorentzAngleWidth, 0),
      edgeClusterErrorX_(conf.getParameter<double>("EdgeClusterErrorX")),
      edgeClusterErrorY_(conf.getParameter<double>("EdgeClusterErrorY")),
      useErrorsFromTemplates_(conf.getParameter<bool>("UseErrorsFromTemplates")),
      truncatePixelCharge_(conf.getParameter<bool>("TruncatePixelCharge")),
      xerr_barrel_l1_{conf.getParameter<std::vector<double>>("xerr_barrel_l1")},
      yerr_barrel_l1_{conf.getParameter<std::vector<double>>("yerr_barrel_l1")},
      xerr_barrel_ln_{conf.getParameter<std::vector<double>>("xerr_barrel_ln")},
      yerr_barrel_ln_{conf.getParameter<std::vector<double>>("yerr_barrel_ln")},
      xerr_endcap_{conf.getParameter<std::vector<double>>("xerr_endcap")},
      yerr_endcap_{conf.getParameter<std::vector<double>>("yerr_endcap")},
      xerr_barrel_l1_def_{conf.getParameter<double>("xerr_barrel_l1_def")},
      yerr_barrel_l1_def_{conf.getParameter<double>("yerr_barrel_l1_def")},
      xerr_barrel_ln_def_{conf.getParameter<double>("xerr_barrel_ln_def")},
      yerr_barrel_ln_def_{conf.getParameter<double>("yerr_barrel_ln_def")},
      xerr_endcap_def_{conf.getParameter<double>("xerr_endcap_def")},
      yerr_endcap_def_{conf.getParameter<double>("yerr_endcap_def")} {};

std::unique_ptr<PixelCPEBase::ClusterParam> PixelCPEGenericBase::createClusterParam(const SiPixelCluster& cl) const {
  return std::make_unique<ClusterParamGeneric>(cl);
}

//-----------------------------------------------------------------------------
//!  Collect the edge charges in x and y, in a single pass over the pixel vector.
//!  Calculate charge in the first and last pixel projected in x and y
//!  and the inner cluster charge, projected in x and y.
//-----------------------------------------------------------------------------
void PixelCPEGenericBase::collect_edge_charges(ClusterParam& theClusterParamBase,  //!< input, the cluster
                                               int& q_f_X,                         //!< output, Q first  in X
                                               int& q_l_X,                         //!< output, Q last   in X
                                               int& q_f_Y,                         //!< output, Q first  in Y
                                               int& q_l_Y,                         //!< output, Q last   in Y
                                               bool truncate) {
  ClusterParamGeneric& theClusterParam = static_cast<ClusterParamGeneric&>(theClusterParamBase);

  // Initialize return variables.
  q_f_X = q_l_X = 0;
  q_f_Y = q_l_Y = 0;

  // Obtain boundaries in index units
  int xmin = theClusterParam.theCluster->minPixelRow();
  int xmax = theClusterParam.theCluster->maxPixelRow();
  int ymin = theClusterParam.theCluster->minPixelCol();
  int ymax = theClusterParam.theCluster->maxPixelCol();

  // Iterate over the pixels.
  int isize = theClusterParam.theCluster->size();
  for (int i = 0; i != isize; ++i) {
    auto const& pixel = theClusterParam.theCluster->pixel(i);
    // ggiurgiu@fnal.gov: add pixel charge truncation
    int pix_adc = pixel.adc;
    if (truncate)
      pix_adc = std::min(pix_adc, theClusterParam.pixmx);

    //
    // X projection
    if (pixel.x == xmin)
      q_f_X += pix_adc;
    if (pixel.x == xmax)
      q_l_X += pix_adc;
    //
    // Y projection
    if (pixel.y == ymin)
      q_f_Y += pix_adc;
    if (pixel.y == ymax)
      q_l_Y += pix_adc;
  }
}

void PixelCPEGenericBase::fillPSetDescription(edm::ParameterSetDescription& desc) {
  desc.add<std::vector<double>>("xerr_barrel_l1", {0.00115, 0.00120, 0.00088});
  desc.add<std::vector<double>>("yerr_barrel_l1",
                                {0.00375, 0.00230, 0.00250, 0.00250, 0.00230, 0.00230, 0.00210, 0.00210, 0.00240});
  desc.add<std::vector<double>>("xerr_barrel_ln", {0.00115, 0.00120, 0.00088});
  desc.add<std::vector<double>>("yerr_barrel_ln",
                                {0.00375, 0.00230, 0.00250, 0.00250, 0.00230, 0.00230, 0.00210, 0.00210, 0.00240});
  desc.add<std::vector<double>>("xerr_endcap", {0.0020, 0.0020});
  desc.add<std::vector<double>>("yerr_endcap", {0.00210});
  desc.add<double>("xerr_barrel_l1_def", 0.01030);
  desc.add<double>("yerr_barrel_l1_def", 0.00210);
  desc.add<double>("xerr_barrel_ln_def", 0.01030);
  desc.add<double>("yerr_barrel_ln_def", 0.00210);
  desc.add<double>("xerr_endcap_def", 0.0020);
  desc.add<double>("yerr_endcap_def", 0.00075);
}
