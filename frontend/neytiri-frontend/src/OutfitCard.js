import React from "react";
import "./OutfitCard.css";

const OutfitCard = ({ outfit, id }) => {
  var articleType = "Not Defined";
  switch (id) {
    case 0:
      articleType = "Top Wear";
      // code block
      break;
    case 1:
      articleType = "Bottom Wear";
      // code block
      break;
    case 2:
      articleType = "Foot Wear";
      // code block
      break;
    case 3:
      articleType = "Accessories";
      break;
  }
  return (
    console.log(outfit),
    <div className="CardWrapper">
      <img
        src={outfit.link}
        alt={outfit.product_display_name}
        className="CardImage"
      />
      <div className="TextsHolder">
        <p className="ProductName">{outfit.product_display_name}</p>
        <p className="OtherName">Brand: {outfit.brand_name}</p>
        <p className="OtherName">Color: {outfit.color}</p>

        {/* <p className="OtherName">MasterCategory: {outfit.master_category}</p> */}
        {/* <p className="OtherName">SubCategory: {outfit.sub_category}</p> */}
        <p className="OtherName">ArticleType: {outfit.article_type}</p>
        {/* <p className="Link">Link: {outfit.link}</p> */}
        <a href={outfit.link} className="Link" target="_blank" rel="noopener noreferrer"> Image Link</a>
        {/* <p className="OtherName" style={{ color: "red " }}>
          HardCodedType: {articleType}
        </p> */}
      </div>
    </div>
  );
};

export default OutfitCard;
